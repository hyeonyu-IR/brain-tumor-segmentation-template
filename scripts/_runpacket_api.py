# scripts/_runpacket_api.py
from __future__ import annotations

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


RE_TRAIN_LOSS = re.compile(r"Epoch\s+(\d+)\s+train_loss=([0-9]*\.?[0-9]+)")
RE_VAL_DICE = re.compile(
    r"Epoch\s+(\d+)\s+val_dice\s+WT=([0-9]*\.?[0-9]+)\s+TC=([0-9]*\.?[0-9]+)\s+ET=([0-9]*\.?[0-9]+)\s+mean=([0-9]*\.?[0-9]+)"
)

def _parse_log(log_path: Path):
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
    return {
        "epochs": epochs,
        "train_loss": [train.get(e, np.nan) for e in epochs],
        "WT": [val.get(e, {}).get("WT", np.nan) for e in epochs],
        "TC": [val.get(e, {}).get("TC", np.nan) for e in epochs],
        "ET": [val.get(e, {}).get("ET", np.nan) for e in epochs],
        "mean": [val.get(e, {}).get("mean", np.nan) for e in epochs],
        "raw_text": text,
    }

def _save_curves(out_dir: Path, ts: str, curves: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_png = out_dir / f"curves_{ts}_loss.png"
    dice_png = out_dir / f"curves_{ts}_dice.png"

    epochs = curves["epochs"]
    if not epochs:
        raise RuntimeError("No epochs found in run log; cannot plot curves.")

    plt.figure()
    plt.plot(epochs, curves["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=160)
    plt.close()

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
    plt.savefig(dice_png, dpi=160)
    plt.close()

    return loss_png, dice_png

def _wrap(text: str, width: int = 110):
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

def _draw_heading(c, x, y, s, size=14):
    c.setFont("Helvetica-Bold", size)
    c.drawString(x, y, s)

def _draw_body(c, x, y, lines, size=9, leading=11, bottom=0.75 * inch):
    c.setFont("Helvetica", size)
    for line in lines:
        if y < bottom:
            c.showPage()
            c.setFont("Helvetica", size)
            y = letter[1] - 0.75 * inch
        c.drawString(x, y, line)
        y -= leading
    return y

def _draw_image_fit(c, img_path: Path, x, y, max_w, max_h, caption=None):
    img = ImageReader(str(img_path))
    iw, ih = img.getSize()
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    c.drawImage(img, x, y - h, width=w, height=h, preserveAspectRatio=True, mask="auto")
    if caption:
        c.setFont("Helvetica", 9)
        c.drawString(x, y - h - 12, caption)
        return y - h - 24
    return y - h - 12


def make_runpacket_pdf(
    *,
    config_path: Path,
    run_log_path: Path,
    report_md_path: Path,
    eval_summary_path: Path,
    fig_dir: Path,
    out_dir: Path,
    max_figs: int = 6,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"runpacket_{ts}.pdf"

    cfg = yaml.safe_load(config_path.read_text())
    curves = _parse_log(run_log_path)
    loss_png, dice_png = _save_curves(out_dir, ts, curves)

    md_text = report_md_path.read_text(encoding="utf-8", errors="ignore") if report_md_path.exists() else ""
    md_lines = _wrap(md_text, width=115)

    summary = json.loads(eval_summary_path.read_text()) if eval_summary_path.exists() else None

    figs = []
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))[:max_figs]

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    W, H = letter
    x0 = 0.75 * inch
    y = H - 0.75 * inch

    _draw_heading(c, x0, y, "Brain Tumor Segmentation — Run Packet", size=16)
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(x0, y, f"Generated: {ts}"); y -= 14
    c.drawString(x0, y, f"Config: {config_path.as_posix()}"); y -= 14
    c.drawString(x0, y, f"Run log: {run_log_path.as_posix()}"); y -= 20

    _draw_heading(c, x0, y, "Hyperparameters (from YAML)", size=13)
    y -= 18
    cfg_dump = yaml.safe_dump(cfg, sort_keys=True).strip()
    y = _draw_body(c, x0, y, _wrap(cfg_dump, width=115), size=8, leading=10)

    c.showPage()
    y = H - 0.75 * inch

    _draw_heading(c, x0, y, "Evaluation Summary", size=13)
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
    y = _draw_body(c, x0, y, lines, size=10, leading=13)

    y -= 10
    _draw_heading(c, x0, y, "Training/Validation Curves", size=13)
    y -= 18
    y = _draw_image_fit(c, loss_png, x0, y, max_w=W - 1.5 * inch, max_h=3.0 * inch, caption="Train loss vs epoch")
    y -= 10
    y = _draw_image_fit(c, dice_png, x0, y, max_w=W - 1.5 * inch, max_h=3.0 * inch, caption="Validation Dice vs epoch")

    c.showPage()
    y = H - 0.75 * inch

    _draw_heading(c, x0, y, "Representative Overlays", size=13)
    y -= 18
    if not figs:
        y = _draw_body(c, x0, y, ["No figures found under eval/figures (skipped)."], size=10, leading=13)
    else:
        for fp in figs:
            if y < 2.0 * inch:
                c.showPage()
                y = H - 0.75 * inch
            y = _draw_image_fit(c, fp, x0, y, max_w=W - 1.5 * inch, max_h=3.2 * inch, caption=fp.name)
            y -= 8

    c.showPage()
    y = H - 0.75 * inch
    _draw_heading(c, x0, y, "report.md (rendered as text)", size=13)
    y -= 18
    if md_lines:
        y = _draw_body(c, x0, y, md_lines, size=8, leading=10)
    else:
        y = _draw_body(c, x0, y, ["No report.md found (skipped)."], size=10, leading=13)

    c.showPage()
    y = H - 0.75 * inch
    _draw_heading(c, x0, y, "Run Log Tail (last 200 lines)", size=13)
    y -= 18
    tail_lines = curves["raw_text"].splitlines()[-200:]
    y = _draw_body(c, x0, y, _wrap("\n".join(tail_lines), width=115), size=7, leading=9)

    c.save()
    return pdf_path
