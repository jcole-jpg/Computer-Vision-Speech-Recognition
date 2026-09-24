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

# Paths come from the single config module - never re-derive them here (B0).
from .config import (  # noqa: F401  (re-exported for backwards compatibility)
    DATA_DIR,
    FIG_DIR,
    IMAGE_DIR,
    PROJECT_ROOT,
)

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


def build_lesion_table(images: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per lesion, in the exact schema the project pipeline splits on (A1.3).

    Columns: ``lesion_id, derm_id, clinical_id, diagnosis_1, dx, age, sex, site``
    where ``dx`` is the 11-class label from ``training_gt.csv`` and ``derm_id`` /
    ``clinical_id`` are the ``isic_id`` of the dermoscopic / clinical view.

    Built with pivot + groupby only - no Python loop over rows - because the
    per-lesion fields were verified identical across a lesion's two images
    (A1.1e), so ``first`` is a safe aggregation.
    """
    df = images if images is not None else load_images_table()

    # Wide: one column per image_type holding that view's isic_id.
    ids = (
        df.pivot_table(index="lesion_id", columns="image_type",
                       values="isic_id", aggfunc="first")
        .rename(columns={"dermoscopic": "derm_id", "clinical": "clinical_id"})
    )
    ids.columns.name = None

    # Per-lesion fields: identical across the two images, so take the first.
    fields = df.groupby("lesion_id").agg(
        diagnosis_1=("diagnosis_1", "first"),
        dx=("label", "first"),
        age=("age_approx", "first"),
        sex=("sex", "first"),
        site=("anatom_site_general", "first"),
    )

    out = ids.join(fields).reset_index()
    return out[["lesion_id", "derm_id", "clinical_id",
                "diagnosis_1", "dx", "age", "sex", "site"]]


def assert_lesion_table(lesions: pd.DataFrame, n_expected: int = 5_240) -> None:
    """The A1.3 acceptance checks, as asserts (raises with a useful message)."""
    assert len(lesions) == n_expected, f"expected {n_expected} lesions, got {len(lesions)}"
    assert lesions["lesion_id"].is_unique, "duplicate lesion_id"
    for col in ("derm_id", "clinical_id"):
        missing = lesions[col].isna().sum()
        assert missing == 0, f"{missing} lesions are missing {col}"


#: ``site`` (training_input.csv) -> the vocabulary used by ``anatom_site_general``.
SITE_ALIASES: dict[str, str] = {
    "trunk": "trunk",
    "head_neck_face": "head/neck",
    "upper_extremity": "upper extremity",
    "lower_extremity": "lower extremity",
    "hand": "upper extremity",
    "foot": "lower extremity",
    "genital": "oral/genital",
}


def resolve_site(images: pd.DataFrame) -> pd.Series:
    """Recover the true anatomical site, filling ``anatom_site_general`` from ``site``.

    ``anatom_site_general`` has **no 'trunk' category**: trunk lesions are recorded
    as NaN there, and the real value lives in the ``site`` column of
    ``training_input.csv``. 3,850 of the 3,912 "missing" images are trunk, and only
    62 are genuinely unknown.

    Imputing the raw column with "unknown" would therefore collapse the single most
    common anatomical site into a meaningless bucket and throw away real signal.
    Use this instead; it leaves only the 62 true unknowns.
    """
    filled = images["anatom_site_general"].copy()
    if "site" not in images.columns:
        return filled.fillna("unknown")
    recovered = images["site"].map(SITE_ALIASES)
    return filled.fillna(recovered).fillna("unknown")
