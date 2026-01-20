# Brain Tumor Segmentation Template (MSD Task01 / BraTS-style)

Baseline 3D multi-modal MRI brain tumor segmentation using MONAI.

## Dataset
Expected dataset layout (not committed to git):
- imagesTr/ (training images, NIfTI .nii.gz, shape H×W×D×4)
- labelsTr/ (training labels, NIfTI .nii.gz, shape H×W×D)
- imagesTs/ (test images, no labels)
- dataset.json (authoritative modality + label mapping)

From `dataset.json`:
- Modalities: 0=FLAIR, 1=T1w, 2=t1gd, 3=T2w
- Labels: 0=background, 1=edema, 2=non-enhancing tumor, 3=enhancing tumour

## Quickstart
1) Create split:
```
python scripts/01_make_splits_from_dataset_json.py --data_dir <DATASET_ROOT>
```

2) Train baseline:
```
python -m src.train --config configs/baseline_3d_unet.yaml
```
3) Metrics
BraTS-style region Dice:
- WT = {1,2,3}
- TC = {2,3}
- ET = {3}

4) Evaluation
```
python -m scripts.02_eval --config configs\baseline_3d_unet.yaml --num_vis 8
```

4) Report generation
```
python -m scripts.03_make_report --eval_dir outputs\baseline_3d_unet\eval --out outputs\baseline_3d_unet\report.md
```
5) PDF file generation
```
python -m scripts.05_make_runpacket
```

