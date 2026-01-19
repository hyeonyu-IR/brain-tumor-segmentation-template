# src/metrics.py
# per-class Dice + WT/TC/ET

import torch

def dice_per_class(pred_onehot: torch.Tensor, true_onehot: torch.Tensor, eps: float = 1e-6):
    """
    pred_onehot, true_onehot: (B, C, H, W, D) bool/float
    returns: (C,) dice averaged over batch
    """
    assert pred_onehot.shape == true_onehot.shape
    B, C = pred_onehot.shape[:2]
    dices = []
    for c in range(C):
        p = pred_onehot[:, c].reshape(B, -1).float()
        t = true_onehot[:, c].reshape(B, -1).float()
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        d = (2.0 * inter + eps) / (denom + eps)
        dices.append(d.mean())
    return torch.stack(dices, dim=0)  # (C,)

def brats_region_dice(pred_label: torch.Tensor, true_label: torch.Tensor, eps: float = 1e-6):
    """
    pred_label, true_label: (B, H, W, D) int labels in {0,1,2,3}
    Using dataset.json semantics:
      1=edema, 2=non-enhancing tumor, 3=enhancing tumour
    Regions:
      WT = {1,2,3}
      TC = {2,3}
      ET = {3}
    returns dict of scalar dices averaged over batch
    """
    B = pred_label.shape[0]
    def _dice(mask_p, mask_t):
        p = mask_p.reshape(B, -1).float()
        t = mask_t.reshape(B, -1).float()
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        return ((2.0 * inter + eps) / (denom + eps)).mean()

    pred = pred_label
    true = true_label

    wt_p = (pred > 0)
    wt_t = (true > 0)

    tc_p = (pred == 2) | (pred == 3)
    tc_t = (true == 2) | (true == 3)

    et_p = (pred == 3)
    et_t = (true == 3)

    return {
        "WT": _dice(wt_p, wt_t),
        "TC": _dice(tc_p, tc_t),
        "ET": _dice(et_p, et_t),
    }
