# scripts/01_make_splits_from_dataset_json.py
import argparse, json, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Root directory containing dataset.json, imagesTr, labelsTr, imagesTs")
    ap.add_argument("--out_json", type=str, default="splits/split_seed42.json")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds_json = data_dir / "dataset.json"
    if not ds_json.exists():
        raise FileNotFoundError(f"dataset.json not found at: {ds_json}")

    ds = json.loads(ds_json.read_text())
    training = ds.get("training", [])
    if not training:
        raise ValueError("dataset.json has no 'training' entries.")

    pairs = []
    for item in training:
        img_rel = item["image"]
        lbl_rel = item["label"]
        img_path = (data_dir / img_rel).resolve()
        lbl_path = (data_dir / lbl_rel).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {lbl_path}")
        case_id = Path(img_rel).name.replace(".nii.gz", "")
        pairs.append({"id": case_id, "image": str(img_path), "label": str(lbl_path)})

    random.seed(args.seed)
    random.shuffle(pairs)

    n_val = int(round(len(pairs) * args.val_frac))
    val = pairs[:n_val]
    train = pairs[n_val:]

    out = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "train": train,
        "val": val,
        "dataset": {
            "name": ds.get("name"),
            "description": ds.get("description"),
            "modality": ds.get("modality"),
            "labels": ds.get("labels"),
        }
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} (train={len(train)}, val={len(val)})")

if __name__ == "__main__":
    main()
