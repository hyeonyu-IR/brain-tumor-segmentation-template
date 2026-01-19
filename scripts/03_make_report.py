# scripts/03_make_report.py
import argparse
from pathlib import Path
import json
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, default="outputs/baseline_3d_unet/eval")
    ap.add_argument("--out", type=str, default="outputs/baseline_3d_unet/report.md")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    df = pd.read_csv(eval_dir / "val_metrics.csv")
    summary = json.loads((eval_dir / "summary.json").read_text())

    # pick top/bottom examples for links
    df_sorted = df.sort_values("mean", ascending=False)
    top = df_sorted.head(3)
    bottom = df_sorted.tail(3)

    lines = []
    lines.append("# Brain Tumor Segmentation — Baseline Report\n")
    lines.append("## Model\n")
    lines.append("- Architecture: MONAI 3D UNet\n")
    lines.append(f"- Checkpoint: `{summary['checkpoint']}`\n")
    lines.append(f"- Saved epoch: {summary['epoch']}\n")
    lines.append("\n## Validation metrics (Dice)\n")
    lines.append(f"- WT:  {summary['WT_mean']:.4f} ± {summary['WT_std']:.4f}\n")
    lines.append(f"- TC:  {summary['TC_mean']:.4f} ± {summary['TC_std']:.4f}\n")
    lines.append(f"- ET:  {summary['ET_mean']:.4f} ± {summary['ET_std']:.4f}\n")
    lines.append(f"- Mean:{summary['mean_mean']:.4f} ± {summary['mean_std']:.4f}\n")

    lines.append("\n## Qualitative examples\n")
    lines.append("Figures are overlays on FLAIR with GT and Pred.\n")

    lines.append("\n### Top 3 cases by mean Dice\n")
    for _, r in top.iterrows():
        fig = eval_dir / "figures" / f"{r['case_id']}.png"
        rel = fig.as_posix()
        lines.append(f"- **{r['case_id']}** (mean={r['mean']:.3f}, WT={r['WT']:.3f}, TC={r['TC']:.3f}, ET={r['ET']:.3f})\n")
        if fig.exists():
            lines.append(f"  ![]({rel})\n")

    lines.append("\n### Bottom 3 cases by mean Dice\n")
    for _, r in bottom.iterrows():
        fig = eval_dir / "figures" / f"{r['case_id']}.png"
        rel = fig.as_posix()
        lines.append(f"- **{r['case_id']}** (mean={r['mean']:.3f}, WT={r['WT']:.3f}, TC={r['TC']:.3f}, ET={r['ET']:.3f})\n")
        if fig.exists():
            lines.append(f"  ![]({rel})\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print("Wrote report:", out_path)


if __name__ == "__main__":
    main()
