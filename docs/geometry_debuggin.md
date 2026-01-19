# BraTS Geometry Debugging Guide

This document records the full debugging process that was required to make the Medical Segmentation Decathlon (MSD) BraTS dataset train correctly with MONAI. This problem is extremely common in 3D multi‑modal medical imaging and is a major source of silent errors in research pipelines.

This guide is written so that future users (and future us) do not have to rediscover these failures.

---

## 1. The Root Cause

The BraTS dataset has an unusual but very important structure:

| Component | Raw shape    |
| --------- | ------------ |
| Image     | (H, W, D, 4) |
| Label     | (H, W, D)    |

MONAI assumes channel‑first tensors by default:

| Expected by MONAI   |
| ------------------- |
| Image: (C, H, W, D) |
| Label: (1, H, W, D) |

If this is not enforced explicitly, MONAI misinterprets axes:

* A 3D label `(H,W,D)` is interpreted as `(C,H,W)`
* A 4D image `(H,W,D,4)` is interpreted ambiguously

This causes:

* Orientationd warnings: `D=2`
* Spacingd crashes: `(3,) vs (2,)`
* DiceCELoss failures due to dimension mismatch

These are not bugs in MONAI — they are consequences of ambiguous tensor geometry.

---

## 2. Mandatory Geometry Rules for BraTS

Before any spatial transform:

| Tensor | Must be converted to |
| ------ | -------------------- |
| Image  | (4, H, W, D)         |
| Label  | (1, H, W, D)         |

This must occur **before** Orientationd or Spacingd.

---

## 3. Final Working Strategy

We solved this by enforcing geometry explicitly using `Lambdad`.

### 3.1 Image channel fix

```python
def _force_channel_first_img(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got {x.shape}")

    # (H,W,D,4) -> (4,H,W,D)
    if x.shape[-1] == 4:
        x = x.permute(3, 0, 1, 2).contiguous()
    elif x.shape[0] == 4:
        x = x.contiguous()
    else:
        raise ValueError(f"Unexpected image shape {x.shape}")

    return x
```

### 3.2 Label channel fix

```python
def _ensure_label_b1hwd(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 4 and y.shape[-1] == 1:
        y = y[..., 0]

    if y.ndim == 3:
        y = y.unsqueeze(0)   # (1,H,W,D)
    elif y.ndim == 4 and y.shape[0] == 1:
        pass
    else:
        raise ValueError(f"Unexpected label shape {y.shape}")

    return y.long()
```

---

## 4. Correct Transform Ordering (Critical)

The transform pipeline **must** follow this order:

```python
LoadImaged(keys=["image","label"]),
EnsureTyped(keys=["image","label"]),

Lambdad(keys=["image"], func=_force_channel_first_img),
Lambdad(keys=["label"], func=_ensure_label_b1hwd),

Orientationd(keys=["image","label"], axcodes="RAS"),
Spacingd(keys=["image","label"], pixdim=(1,1,1)),
```

If `Spacingd` runs before label channelization, the pipeline will crash.

---

## 5. Why DiceCELoss Failed

Model output:

```
(B, 4, H, W, D)
```

Correct target format:

```
(B, 1, H, W, D)
```

If label is `(B,H,W,D)`, DiceCELoss throws:

```
input dim=5, target dim=4
```

Therefore:

* Do NOT squeeze label before loss
* Squeeze only for metrics

---

## 6. Correct Training Loop Usage

```python
lbl = batch["label"].to(device)   # (B,1,H,W,D)
loss = loss_fn(logits, lbl)
```

Metrics:

```python
lab = lbl.squeeze(1)               # (B,H,W,D)
```

---

## 7. Debugging Methodology (Reusable Pattern)

Key techniques used:

1. Bypass CLI with `%run scripts/99_debug_train.py`
2. Print tensor shapes inside transforms
3. Restart kernel after every geometry change
4. Disable caching/workers while debugging

This methodology applies to all 3D medical segmentation projects.

---

## 8. Why This Matters Scientifically

Incorrect geometry does not always crash — it often trains silently on corrupted spatial structure. Many published models suffer from hidden orientation/channel bugs.

A pipeline that enforces geometry explicitly is:

* Reproducible
* Auditable
* Scientifically defensible

---

## 9. Recommended Git Tag

Once stable:

```bash
git tag v0.1-geometry-stable
```

This preserves the hardest part of the project permanently.

---

This document should be kept in `docs/geometry_debugging.md` for all future segmentation projects.
