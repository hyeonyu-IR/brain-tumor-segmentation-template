# scripts/02_eval.py
import argparse
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from src.data.msd_brain import get_loaders
from src.metrics import brats_region_dice


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=cfg["data"]["in_channels"],
        out_channels=cfg["data"]["num_classes"],
        channels=cfg["model"]["channels"],
        strides=cfg["model"]["strides"],
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt


def pick_slices(label_hwd: np.ndarray) -> list[int]:
    """
    Choose 3 z-slices for visualization:
    - center slice
    - slice of max tumor area
    - slice of max enhancing tumor area (if present)
    """
    zc = label_hwd.shape[-1] // 2

    tumor = (label_hwd > 0).astype(np.uint8)
    area = tumor.sum(axis=(0, 1))  # per-z
    z_tumor = int(area.argmax()) if area.max() > 0 else zc

    et = (label_hwd == 3).astype(np.uint8)
    area_et = et.sum(axis=(0, 1))
    z_et = int(area_et.argmax()) if area_et.max() > 0 else z_tumor

    # unique and sorted
    zs = sorted(list({zc, z_tumor, z_et}))
    return zs


def overlay_seg(ax, seg2d: np.ndarray, alpha: float = 0.35):
    """
    Overlay segmentation mask with fixed class colors (no external dependencies).
    0=bg, 1,2,3 tumor classes.
    """
    # Create RGBA overlay
    h, w = seg2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    # class colors: 1=red, 2=green, 3=blue
    colors = {
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.4, 1.0),
    }
    for k, (r, g, b) in colors.items():
        m = seg2d == k
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = alpha

    ax.imshow(rgba)


def save_case_figure(out_png: Path, img_chwd: np.ndarray, gt_hwd: np.ndarray, pr_hwd: np.ndarray):
    """
    Save a compact qualitative panel for one case.
    img_chwd: (4,H,W,D), gt/pr: (H,W,D)
    """
    zs = pick_slices(gt_hwd)

    # layout: rows = z slices, cols = modalities + (GT overlay) + (Pred overlay)
    # cols: T1, T1c, T2, FLAIR, GT, Pred
    nrows = len(zs)
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["T1", "T1c", "T2", "FLAIR", "GT", "Pred"]

    for ri, z in enumerate(zs):
        # modalities
        for ci in range(4):
            ax = axes[ri, ci]
            ax.imshow(img_chwd[ci, :, :, z], cmap="gray")
            ax.set_title(f"{titles[ci]} (z={z})" if ri == 0 else f"z={z}")
            ax.axis("off")

        # GT overlay on FLAIR (channel 3)
        ax_gt = axes[ri, 4]
        ax_gt.imshow(img_chwd[3, :, :, z], cmap="gray")
        overlay_seg(ax_gt, gt_hwd[:, :, z])
        ax_gt.set_title("GT overlay" if ri == 0 else "")
        ax_gt.axis("off")

        # Pred overlay on FLAIR
        ax_pr = axes[ri, 5]
        ax_pr.imshow(img_chwd[3, :, :, z], cmap="gray")
        overlay_seg(ax_pr, pr_hwd[:, :, z])
        ax_pr.set_title("Pred overlay" if ri == 0 else "")
        ax_pr.axis("off")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt. Default: outputs/baseline_3d_unet/best.pt")
    ap.add_argument("--out_dir", type=str, default="outputs/baseline_3d_unet/eval")
    ap.add_argument("--num_vis", type=int, default=6, help="Number of qualitative cases to save")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # Default checkpoint path
    if args.ckpt is None:
        ckpt_path = Path(cfg["project"]["out_dir"]) / "baseline_3d_unet" / "best.pt"
    else:
        ckpt_path = Path(args.ckpt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    _, val_loader = get_loaders(cfg)

    # Model
    model = build_model(cfg, device)
    ckpt = load_checkpoint(model, ckpt_path, device)

    rows = []
    saved_vis = 0

    for i, batch in enumerate(val_loader):
        img = batch["image"].to(device)     # (B,4,H,W,D)
        lbl = batch["label"].to(device)     # (B,1,H,W,D)
        lab = lbl.squeeze(1)                # (B,H,W,D)

        sw_roi = tuple(cfg["infer"]["sw_roi_size"])
        pred_logits = sliding_window_inference(
            img, sw_roi, cfg["infer"]["sw_batch_size"], model, overlap=cfg["infer"]["overlap"]
        )  # (B,C,H,W,D)

        pred = torch.argmax(pred_logits, dim=1)  # (B,H,W,D)

        # metrics per batch element
        B = pred.shape[0]
        for b in range(B):
            dices = brats_region_dice(pred[b:b+1], lab[b:b+1])
            wt = float(dices["WT"])
            tc = float(dices["TC"])
            et = float(dices["ET"])
            mean = (wt + tc + et) / 3.0

            case_id = batch.get("case_id", [None] * B)[b]
            if case_id is None:
                # fallback to index
                case_id = f"val_{i:04d}_b{b}"

            rows.append(
                {"case_id": str(case_id), "WT": wt, "TC": tc, "ET": et, "mean": mean}
            )

            # Qualitative visualization (first N cases)
            if saved_vis < args.num_vis:
                img_np = img[b].detach().cpu().numpy()        # (4,H,W,D)
                gt_np = lab[b].detach().cpu().numpy()         # (H,W,D)
                pr_np = pred[b].detach().cpu().numpy()        # (H,W,D)

                out_png = out_dir / "figures" / f"{case_id}.png"
                save_case_figure(out_png, img_np, gt_np, pr_np)
                saved_vis += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "val_metrics.csv", index=False)

    summary = {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "metric_saved": float(ckpt.get("metric", np.nan)),
        "n_val_samples": int(len(df)),
        "WT_mean": float(df["WT"].mean()),
        "WT_std": float(df["WT"].std(ddof=1)),
        "TC_mean": float(df["TC"].mean()),
        "TC_std": float(df["TC"].std(ddof=1)),
        "ET_mean": float(df["ET"].mean()),
        "ET_std": float(df["ET"].std(ddof=1)),
        "mean_mean": float(df["mean"].mean()),
        "mean_std": float(df["mean"].std(ddof=1)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved:", out_dir / "val_metrics.csv")
    print("Saved:", out_dir / "summary.json")
    print("Saved figures:", out_dir / "figures")


if __name__ == "__main__":
    main()
