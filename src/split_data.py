import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# ===== Paths (Colab-native) =====
BASE_DIR = Path("/content/plantdisease/PlantVillage")
SPLIT_ROOT = Path("/content/plantdisease_split")

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXTS = {".jpg", ".jpeg", ".png"}

rng = np.random.default_rng(42)

def safe_copy(src, dst):
    if not dst.exists():
        shutil.copy2(src, dst)

def main():
    print("Preparing stratified train/val/test split...")

    classes = sorted([c for c in os.listdir(BASE_DIR) if (BASE_DIR / c).is_dir()])

    # Create directories
    for split in RATIOS:
        for c in classes:
            (SPLIT_ROOT / split / c).mkdir(parents=True, exist_ok=True)

    rows = []

    for c in classes:
        files = [p for p in (BASE_DIR / c).iterdir() if p.suffix.lower() in EXTS]
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * RATIOS["train"])
        n_val   = int(n * RATIOS["val"])

        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:]
        }

        row = {"Class": c}

        for split_name, flist in splits.items():
            for f in flist:
                safe_copy(f, SPLIT_ROOT / split_name / c / f.name)
            row[split_name.capitalize()] = len(flist)

        row["Total"] = n
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)

    print("\nPer-class distribution after split:")
    print(df)

    print("\nTotal images per split:")
    print(df[["Train", "Val", "Test"]].sum())

    print("\nDataset split completed successfully.")

if __name__ == "__main__":
    main()
