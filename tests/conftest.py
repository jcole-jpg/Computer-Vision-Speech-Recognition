"""Shared fixtures: a tiny synthetic image folder + metadata table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import milk10k  # noqa: E402


@pytest.fixture
def synthetic_dataset(tmp_path):
    """6 images (3 classes x 2), one corrupt file, one metadata row with no file."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    sizes = [(640, 480), (480, 640), (300, 300), (1024, 768), (200, 150), (600, 450)]
    labels = ["NV", "BCC", "MEL"] * 2
    for i, (size, label) in enumerate(zip(sizes, labels)):
        isic = f"ISIC_{i:07d}"
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{isic}.jpg", quality=90)
        rows.append({"isic_id": isic, "label": label, "path": img_dir / f"{isic}.jpg"})
    # corrupt file
    (img_dir / "ISIC_BAD.jpg").write_bytes(b"not a jpeg")
    rows.append({"isic_id": "ISIC_BAD", "label": "NV", "path": img_dir / "ISIC_BAD.jpg"})
    # metadata row with no file on disk
    rows.append({"isic_id": "ISIC_MISSING", "label": "NV", "path": img_dir / "ISIC_MISSING.jpg"})
    return img_dir, pd.DataFrame(rows)


@pytest.fixture
def real_images():
    """A few real MILK10k images if the dataset is extracted, else skip."""
    if not milk10k.IMAGE_DIR.exists():
        pytest.skip("MILK10k images not available locally")
    paths = sorted(milk10k.IMAGE_DIR.glob("ISIC_*.jpg"))[:4]
    if not paths:
        pytest.skip("no images found")
    return paths
