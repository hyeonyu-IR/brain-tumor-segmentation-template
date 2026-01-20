# scripts/05_make_runpacket.py
import argparse
from pathlib import Path
import re

def find_latest_run_log(run_dir: Path) -> Path | None:
    """
    Find the most recent run log under outputs/baseline_3d_unet.
    Supports:
      - run_YYYYMMDD_HHMMSS.log
      - run_YYYYMMDD_HHMMSS/log  (folder + file named 'log')
    """
    # 1) *.log files
    logs = sorted(run_dir.glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        return logs[0]

    # 2) run_* folders with a file named 'log' or '*.log' inside
    folders = sorted([p for p in run_dir.glob("run_*") if p.is_dir()],
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for fd in folders:
        candidate = fd / "log"
        if candidate.exists() and candidate.is_file():
            return candidate
        inner_logs = sorted(fd.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if inner_logs:
            return inner_logs[0]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_3d_unet.yaml")
    ap.add_argument("--run_dir", default="outputs/baseline_3d_unet", help="Directory containing run logs and outputs")
    ap.add_argument("--run_log", default=None, help="Optional explicit path to run log. If omitted, auto-detect latest.")
    ap.add_argument("--report_md", default="outputs/baseline_3d_unet/report.md")
    ap.add_argument("--eval_summary", default="outputs/baseline_3d_unet/eval/summary.json")
    ap.add_argument("--fig_dir", default="outputs/baseline_3d_unet/eval/figures")
    ap.add_argument("--out_dir", default="outputs/baseline_3d_unet/reports")
    ap.add_argument("--max_figs", type=int, default=6)
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    cfg_path = (proj_root / args.config).resolve()

    run_dir = (proj_root / args.run_dir).resolve()
    out_dir = (proj_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_log:
        run_log = (proj_root / args.run_log).resolve()
    else:
        run_log = find_latest_run_log(run_dir)
        if run_log is None:
            raise FileNotFoundError(
                f"Could not auto-detect a run log in {run_dir}. "
                "Provide --run_log explicitly."
            )

    report_md = (proj_root / args.report_md).resolve()
    eval_summary = (proj_root / args.eval_summary).resolve()
    fig_dir = (proj_root / args.fig_dir).resolve()

    # Import here so wrapper stays lightweight
    from scripts._runpacket_api import make_runpacket_pdf

    pdf_path = make_runpacket_pdf(
        config_path=cfg_path,
        run_log_path=run_log,
        report_md_path=report_md,
        eval_summary_path=eval_summary,
        fig_dir=fig_dir,
        out_dir=out_dir,
        max_figs=args.max_figs,
    )

    print("Done.")
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
