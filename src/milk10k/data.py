"""Loading helpers and constants for the MILK10k dataset (tables, paths, class map).

Expected layout (git-ignored):
    data/raw/milk10k/
        images/ISIC_*.jpg
        metadata.csv
        supplements/training_input.csv
        supplements/training_gt.csv
        supplements/training_supp.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "milk10k"
IMAGE_DIR = DATA_DIR / "images"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# 11 top-level MILK10k classes (column order of training_gt.csv)
CLASS_NAMES: dict[str, str] = {
    "AKIEC": "Actinic keratosis / intraepidermal carcinoma",
    "BCC": "Basal cell carcinoma",
    "BEN_OTH": "Other benign proliferations",
    "BKL": "Benign keratinocytic lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / infectious",
    "MAL_OTH": "Other malignant proliferations",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "VASC": "Vascular lesion / hemorrhage",
}
CLASS_CODES: list[str] = list(CLASS_NAMES)
MALIGNANT: frozenset[str] = frozenset({"AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"})

SKIN_TONE_LABELS: dict[int, str] = {
    0: "0 very dark", 1: "1 dark", 2: "2 medium-dark",
    3: "3 medium-light", 4: "4 light", 5: "5 very light",
}

# Chart palette (validated: blue/orange/aqua pass CVD + normal-vision checks)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BENIGN_COLOR, MALIGNANT_COLOR = BLUE, ORANGE
SEQ_CMAP = "Blues"


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "metadata.csv")


def load_training_input() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "supplements" / "training_input.csv")


def load_training_gt() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "supplements" / "training_gt.csv")


def load_training_supp() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "supplements" / "training_supp.csv")


def lesion_labels() -> pd.DataFrame:
    """One row per lesion: lesion_id, label (class code), label_name, malignant."""
    gt = load_training_gt()
    onehot = gt[CLASS_CODES]
    assert (onehot.sum(axis=1) == 1).all(), "each lesion must have exactly one class"
    out = pd.DataFrame({
        "lesion_id": gt["lesion_id"],
        "label": onehot.idxmax(axis=1),
    })
    out["label_name"] = out["label"].map(CLASS_NAMES)
    out["malignant"] = out["label"].isin(MALIGNANT)
    return out


def load_images_table() -> pd.DataFrame:
    """One row per image: metadata + training_input + supp + lesion label."""
    meta = load_metadata()
    inp = load_training_input().drop(
        columns=["lesion_id", "image_type", "attribution", "copyright_license",
                 "image_manipulation", "age_approx", "sex"]
    )
    supp = load_training_supp().drop(columns=["diagnosis_confirm_type"])
    df = meta.merge(inp, on="isic_id", how="left").merge(supp, on="isic_id", how="left")
    df = df.merge(lesion_labels(), on="lesion_id", how="left")
    df["image_type"] = df["image_type"].replace({"clinical: close-up": "clinical"})
    df["path"] = df["isic_id"].map(lambda i: IMAGE_DIR / f"{i}.jpg")
    return df


def load_lesions_table(images: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per lesion with patient-level metadata and both image ids."""
    df = images if images is not None else load_images_table()
    first = (
        df.sort_values("image_type")
        .groupby("lesion_id", as_index=False)
        .agg(
            age_approx=("age_approx", "first"),
            sex=("sex", "first"),
            anatom_site_general=("anatom_site_general", "first"),
            site=("site", "first"),
            skin_tone_class=("skin_tone_class", "first"),
            diagnosis_confirm_type=("diagnosis_confirm_type", "first"),
            diagnosis_full=("diagnosis_full", "first"),
            label=("label", "first"),
            label_name=("label_name", "first"),
            malignant=("malignant", "first"),
            n_images=("isic_id", "size"),
        )
    )
    pivot = df.pivot_table(index="lesion_id", columns="image_type",
                           values="isic_id", aggfunc="first")
    pivot.columns = [f"isic_id_{c}" for c in pivot.columns]
    return first.merge(pivot, on="lesion_id", how="left")


def available_subset(df: pd.DataFrame, image_dir: Path | None = None,
                     path_col: str = "path") -> pd.DataFrame:
    """Keep only rows whose image file actually exists on disk.

    Never assume every metadata row has a matching image: partial downloads,
    corrupt extractions or a sub-sampled local copy are all common. ``df`` needs
    a ``path`` column (as produced by :func:`load_images_table`) or an
    ``isic_id`` column, in which case paths are built under ``image_dir``.
    """
    if path_col not in df.columns:
        image_dir = Path(image_dir) if image_dir is not None else IMAGE_DIR
        df = df.assign(**{path_col: df["isic_id"].map(lambda i: image_dir / f"{i}.jpg")})
    exists = df[path_col].map(lambda p: Path(p).is_file())
    return df[exists].reset_index(drop=True)
