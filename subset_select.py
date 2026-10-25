#!/usr/bin/env python3
"""Select the most useful 5% of ID images (stratified, sharpness-ranked).

Criterion: Laplacian variance (focus measure) per image - sharper images
carry more anatomical detail for representation learning. Top 5% per class
are COPIED to a subset tree mirroring the expected folder layout, plus a
manifest CSV with scores. No existing files are touched.
"""
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SRC = Path("Datasets/CXR/TB_Chest_Radiography_Database")
DST = Path("Datasets/CXR_subset_5pct/TB_Chest_Radiography_Database")
FRACTION = 0.05
EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def sharpness_pil(path):
    img = Image.open(path).convert("L").resize((256, 256))
    edge = img.filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(edge, dtype=np.float32).var())


def main():
    rows = []
    for cls in ("Normal", "Tuberculosis"):
        files = [f for f in (SRC / cls).iterdir() if f.suffix in EXTS]
        scored = sorted(((sharpness_pil(f), f) for f in files), reverse=True)
        n = max(1, int(len(scored) * FRACTION))
        print(f"{cls}: {len(scored)} -> top {n} by sharpness")
        for score, f in scored[:n]:
            dest = DST / cls / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            rows.append({"class": cls, "file": f.name, "sharpness": f"{score:.2f}"})
    with open(DST.parent / "subset_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "file", "sharpness"])
        w.writeheader()
        w.writerows(rows)
    print(f"Total: {len(rows)} images -> {DST} + manifest")


if __name__ == "__main__":
    main()
