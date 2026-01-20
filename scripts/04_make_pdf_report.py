# scripts/04_make_pdf_report.py
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -----------------------------
# Parsing utilities
# -----------------------------
RE_TRAIN_LOSS = re.compile(r"Epoch\s+(\d+)\s+train_loss=([0-9]*\.?[0-9]+)")
RE_VAL_DICE = re.compile(
    r"Epoch\s+(\d+)\s+val_dice\s+WT=([0-9]*\.?[0-9]+)\s+TC=([0-9]*\.?[0-9]+)\s+ET=([0-9]*\.?[0-9]+)\s+mean=([0-9]*\.?[0-9]+)"
)

def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    train = {}
    for m in RE_TRAIN_LOSS.finditer(text):
        ep = int(m.group(1))
        train[ep] = float(m.group(2))

    val = {}
    for m in RE_VAL_DICE.finditer(text):
        ep = int(m.group(1))
        val[ep] = {
            "WT": float(m.group(2)),
            "TC": float(m.group(3)),
            "ET": float(m.group(4)),
            "mean": float(m.group(5)),
        }

    epochs = sorted(set(train.keys()) | set(val.keys()))
    train_loss = [train.get(e, np.nan) for e in epochs]
    WT = [val.get(e, {}).get("WT", np.nan) for e in epochs]
    TC = [val.get(e, {}).get("TC", np.nan) for e in epochs]
    ET = [val.get(e, {}).get("ET", np.nan) for e in epochs]
    mean = [val.get(e, {}).get("mean", np.nan) for e in epochs]

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "WT": WT,
        "TC": TC,
        "ET": ET,
        "mean": mean,
        "raw_text": text,
    }


def save_curves_png(curves, out_png: Path):
    epochs = curves["epochs"]
    if not epochs:
        raise RuntimeError("No epochs found in log; cannot plot curves.")

    # Train loss
    plt.figure()
    plt.plot(epochs, curves["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_loss.png"), dpi=160)
    plt.close()

    # Val dice
    plt.figure()
    plt.plot(epochs, curves["WT"], label="WT")
    plt.plot(epochs, curves["TC"], label="TC")
    plt.plot(epochs, curves["ET"], label="ET")
    plt.plot(epochs, curves["mean"], label="mean")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice vs epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_dice.png"), dpi=160)
    plt.close()

    return [
        out_png.with_name(out_png.stem + "_loss.png"),
        out_png.with_name(out_png.stem + "_dice.png"),
    ]


def wrap_text(text: str, width: int = 110):
    # simple wrap for PDF (monospace-ish)
    lines = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
        while len(para) > width:
            lines.append(para[:width])
            para = para[width:]
        lines.append(para)
    return lines


# -----------------------------
# PDF writer helpers
# -----------------------------
def draw_heading(c: canvas.Canvas, x, y, s, size=14):
    c.setFont("Helvetica-Bold", size)
    c.drawString(x, y, s)

def draw_body(c: canvas.Canvas, x, y, lines, size=9, leading=11, bottom_margin=0.75 * inch):
    c.setFont("Helvetica", size)
    for line in lines:
        if y < bottom_margin:
            c.showPage()
            c.setFont("Helvetica", size)
            y = letter[1] - 0.75 * inch
        c.drawString(x, y, line)
        y -= leading
    return y

def draw_image_fit(c: canvas.Canvas, img_path: Path, x, y, max_w, max_h, caption=None):
    img = ImageReader(str(img_path))
    iw, ih = img.getSize()
    scale = min(max_w / iw, max_h / ih)
    w = iw * scale
    h = ih * scale
    c.drawImage(img, x, y - h, width=w, height=h, preserveAspectRatio=True, mask="auto")
    if caption:
        c.setFont("Helvetica", 9)
        c.drawString(x, y - h - 12, caption)
        return y - h - 24
    return y - h - 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/baseline_3d_unet.yaml")
    ap.add_argument("--run_log", required=True, help=r"outputs\baseline_3d_unet\run_YYYYMMDD_HHMMSS.log")
    ap.add_argument("--report_md", default=r"outputs\baseline_3d_unet\report.md")
    ap.add_argument("--eval_summary", default=r"outputs\baseline_3d_unet\eval\summary.json")
    ap.add_argument("--fig_dir", default=r"outputs\baseline_3d_unet\eval\figures")
    ap.add_argument("--out_dir", default=r"outputs\baseline_3d_unet\reports")
    ap.add_argument("--max_figs", type=int, default=6)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    log_path = Path(args.run_log)
    report_md = Path(args.report_md)
    eval_summary = Path(args.eval_summary)
    fig_dir = Path(args.fig_dir)
    out_dir = Path(args.out_dir)

    cfg = yaml.safe_load(cfg_path.read_text())
    curves = parse_log(log_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"runpacket_{ts}.pdf"

    # Prepare plots
    plot_base = out_dir / f"curves_{ts}"
    loss_png, dice_png = save_curves_png(curves, plot_base)

    # Load report.md (optional)
    md_text = report_md.read_text(encoding="utf-8", errors="ignore") if report_md.exists() else ""
    md_lines = wrap_text(md_text, width=115)

    # Load eval summary (optional)
    summary = json.loads(eval_summary.read_text()) if eval_summary.exists() else None

    # Choose representative figures
    figs = []
    if fig_dir.exists():
        # stable ordering
        figs = sorted(fig_dir.glob("*.png"))[: args.max_figs]

    # -----------------------------
    # Build PDF
    # -----------------------------
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    W, H = letter
    x0 = 0.75 * inch
    y = H - 0.75 * inch

    draw_heading(c, x0, y, "Brain Tumor Segmentation — Run Packet", size=16)
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(x0, y, f"Generated: {ts}")
    y -= 14
    c.drawString(x0, y, f"Config: {cfg_path.as_posix()}")
    y -= 14
    c.drawString(x0, y, f"Run log: {log_path.as_posix()}")
    y -= 20

    # Hyperparameters
    draw_heading(c, x0, y, "Hyperparameters (from YAML)", size=13)
    y -= 18
    cfg_dump = yaml.safe_dump(cfg, sort_keys=True).strip()
    y = draw_body(c, x0, y, wrap_text(cfg_dump, width=115), size=8, leading=10)

    c.showPage()
    y = H - 0.75 * inch

    # Summary
    draw_heading(c, x0, y, "Evaluation Summary", size=13)
    y -= 18
    if summary:
        lines = [
            f"Checkpoint: {summary.get('checkpoint', '')}",
            f"Saved epoch: {summary.get('epoch', '')}",
            f"N (val): {summary.get('n_val_samples', '')}",
            f"WT mean±SD:  {summary.get('WT_mean', float('nan')):.4f} ± {summary.get('WT_std', float('nan')):.4f}",
            f"TC mean±SD:  {summary.get('TC_mean', float('nan')):.4f} ± {summary.get('TC_std', float('nan')):.4f}",
            f"ET mean±SD:  {summary.get('ET_mean', float('nan')):.4f} ± {summary.get('ET_std', float('nan')):.4f}",
            f"Mean mean±SD:{summary.get('mean_mean', float('nan')):.4f} ± {summary.get('mean_std', float('nan')):.4f}",
        ]
    else:
        lines = ["No eval summary.json found (skipped)."]
    y = draw_body(c, x0, y, lines, size=10, leading=13)

    y -= 10
    draw_heading(c, x0, y, "Training/Validation Curves", size=13)
    y -= 18

    # Add curve images
    y = draw_image_fit(c, loss_png, x0, y, max_w=W - 1.5 * inch, max_h=3.0 * inch, caption="Train loss vs epoch")
    y -= 10
    y = draw_image_fit(c, dice_png, x0, y, max_w=W - 1.5 * inch, max_h=3.0 * inch, caption="Validation Dice vs epoch")

    c.showPage()
    y = H - 0.75 * inch

    # Figures
    draw_heading(c, x0, y, "Representative Overlays", size=13)
    y -= 18
    if not figs:
        y = draw_body(c, x0, y, ["No figures found under eval/figures (skipped)."], size=10, leading=13)
    else:
        for fp in figs:
            if y < 2.0 * inch:
                c.showPage()
                y = H - 0.75 * inch
            y = draw_image_fit(c, fp, x0, y, max_w=W - 1.5 * inch, max_h=3.2 * inch, caption=fp.name)
            y -= 8

    c.showPage()
    y = H - 0.75 * inch

    # report.md content
    draw_heading(c, x0, y, "report.md (rendered as text)", size=13)
    y -= 18
    if md_lines:
        y = draw_body(c, x0, y, md_lines, size=8, leading=10)
    else:
        y = draw_body(c, x0, y, ["No report.md found (skipped)."], size=10, leading=13)

    # Optionally include the last ~200 lines of the log
    c.showPage()
    y = H - 0.75 * inch
    draw_heading(c, x0, y, "Run Log Tail (last 200 lines)", size=13)
    y -= 18
    tail_lines = curves["raw_text"].splitlines()[-200:]
    y = draw_body(c, x0, y, wrap_text("\n".join(tail_lines), width=115), size=7, leading=9)

    c.save()
    print("Wrote PDF:", pdf_path)


if __name__ == "__main__":
    main()
