# src/train.py
import argparse
from pathlib import Path
import yaml
import torch
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from src.data.msd_brain import get_loaders
from src.metrics import brats_region_dice


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_plain_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    MONAI MetaTensor is a Tensor subclass that can carry metadata.
    For metric computations, it is safer to work with a plain torch.Tensor.
    """
    # MetaTensor has .as_tensor() in MONAI; plain torch.Tensor does not.
    if hasattr(x, "as_tensor"):
        return x.as_tensor()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["project"]["seed"])

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["project"]["out_dir"]) / "baseline_3d_unet"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_loaders(cfg)

    model = UNet(
        spatial_dims=3,
        in_channels=cfg["data"]["in_channels"],
        out_channels=cfg["data"]["num_classes"],
        channels=cfg["model"]["channels"],
        strides=cfg["model"]["strides"],
        num_res_units=cfg["model"]["num_res_units"],
    ).to(device)

    # IMPORTANT:
    # logits: (B,C,H,W,D)
    # target: (B,1,H,W,D) class indices when to_onehot_y=True
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_metric = -1.0
    best_path = out_dir / "best.pt"
    max_epochs = cfg["train"]["max_epochs"]

    for epoch in range(1, max_epochs + 1):

        # =========================
        # Training
        # =========================
        model.train()
        epoch_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{max_epochs} [train]",
            leave=False,
        )

        for batch in train_bar:
            img = batch["image"].to(device)   # (B,4,H,W,D)
            lbl = batch["label"].to(device)   # (B,1,H,W,D)  <-- keep channel dim

            opt.zero_grad(set_to_none=True)

            logits = model(img)               # (B,4,H,W,D)
            loss = loss_fn(logits, lbl)

            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss /= max(1, len(train_loader))
        print(f"Epoch {epoch} train_loss={epoch_loss:.4f}")

        # =========================
        # Validation
        # =========================
        model.eval()
        with torch.no_grad():

            wt_list, tc_list, et_list = [], [], []

            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch} [val]",
                leave=False,
            )

            for batch in val_bar:
                img = batch["image"].to(device)    # (B,4,H,W,D)  typically B=1 in val
                lbl = batch["label"].to(device)    # (B,1,H,W,D)

                sw_roi = tuple(cfg["infer"]["sw_roi_size"])

                pred_logits = sliding_window_inference(
                    img,
                    sw_roi,
                    cfg["infer"]["sw_batch_size"],
                    model,
                    overlap=cfg["infer"]["overlap"],
                )  # (B,C,H,W,D)
                
                pred_label = torch.argmax(pred_logits, dim=1)  # (B,H,W,D)
                lab = lbl.squeeze(1)                            # (B,H,W,D)
                
                # make sure we compute metrics on plain tensors
                if hasattr(pred_label, "as_tensor"):
                    pred_label = pred_label.as_tensor()
                if hasattr(lab, "as_tensor"):
                    lab = lab.as_tensor()
                
                if pred_label.shape != lab.shape:
                    raise RuntimeError(f"Shape mismatch: pred_label {tuple(pred_label.shape)} vs lab {tuple(lab.shape)}")
                
                dices = brats_region_dice(pred_label, lab)


                wt_list.append(float(dices["WT"]))
                tc_list.append(float(dices["TC"]))
                et_list.append(float(dices["ET"]))

                val_bar.set_postfix(
                    {
                        "WT": f"{wt_list[-1]:.3f}",
                        "TC": f"{tc_list[-1]:.3f}",
                        "ET": f"{et_list[-1]:.3f}",
                    }
                )

            WT = sum(wt_list) / len(wt_list)
            TC = sum(tc_list) / len(tc_list)
            ET = sum(et_list) / len(et_list)
            mean_metric = (WT + TC + ET) / 3.0

        print(
            f"Epoch {epoch} val_dice "
            f"WT={WT:.4f} TC={TC:.4f} ET={ET:.4f} mean={mean_metric:.4f}"
        )

        # =========================
        # Checkpointing
        # =========================
        sel = cfg["logging"]["save_best_metric"]
        sel_val = {
            "WT": WT,
            "TC": TC,
            "ET": ET,
            "mean": mean_metric,
        }.get(sel, WT)

        if sel_val > best_metric:
            best_metric = sel_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metric": sel_val,
                },
                best_path,
            )
            print(f"  Saved best checkpoint: {best_path} (metric={best_metric:.4f})")

    print("Training complete.")
    print(f"Best checkpoint: {best_path} (metric={best_metric:.4f})")


if __name__ == "__main__":
    main()
