"""Label strategy: what we predict, and how the strings become integers (B3).

Two decisions are recorded here, both reversible-by-design:

1. **Indeterminate is kept as a third class.** All 123 Indeterminate lesions are
   AKIEC, so the distinction is pathologist confidence rather than a different
   kind of lesion - which is an argument for merging it into Malignant. It is
   kept anyway, because a 3-class model can always be collapsed to the binary
   decision afterwards by summing P(Malignant) + P(Indeterminate), while a model
   trained on merged labels can never be un-merged. Keeping the finer label
   preserves that option at no cost; merging spends it permanently.

2. **All 11 fine classes are kept**, including MAL_OTH (9 lesions). Same
   argument: merging the rare classes into an "other" bucket is irreversible and
   produces a category no clinician can act on. Imbalance is handled where it
   belongs - in the loss weights and the sampler (B8) - not by deleting the
   problem. The honest caveat is that MAL_OTH cannot be *evaluated* reliably at
   9 lesions (~1 lands in test); its per-class metric is reported but never
   trusted, and it is excluded from the headline macro-average.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import CLASS_CODES, CLASS_NAMES, MALIGNANT

#: Primary target: diagnosis_1, three classes, in a fixed order.
DIAGNOSIS1_CLASSES: list[str] = ["Benign", "Indeterminate", "Malignant"]
DIAGNOSIS1_MAP: dict[str, int] = {c: i for i, c in enumerate(DIAGNOSIS1_CLASSES)}

#: Stretch target: the 11-class scheme, in the column order of training_gt.csv.
DX_MAP: dict[str, int] = {c: i for i, c in enumerate(CLASS_CODES)}

#: Documented but NOT used for training - the collapse applied after inference
#: if a binary decision is wanted (see decision 1 above).
BINARY_COLLAPSE: dict[str, str] = {
    "Benign": "Benign", "Indeterminate": "Malignant", "Malignant": "Malignant",
}

#: Classes too rare to evaluate reliably; reported separately, never trusted alone.
LOW_SUPPORT_CLASSES: list[str] = ["MAL_OTH"]


def build_label_map(lesions: pd.DataFrame) -> dict:
    """Assemble the full label map, with the counts that justify each decision."""
    d1_counts = lesions["diagnosis_1"].value_counts().to_dict()
    dx_counts = lesions["dx"].value_counts().to_dict()
    return {
        "primary_target": {
            "column": "diagnosis_1",
            "n_classes": len(DIAGNOSIS1_CLASSES),
            "classes": DIAGNOSIS1_CLASSES,
            "map": DIAGNOSIS1_MAP,
            "lesion_counts": {c: int(d1_counts.get(c, 0)) for c in DIAGNOSIS1_CLASSES},
            "decision": "keep Indeterminate as a third class",
            "rationale": (
                "All Indeterminate lesions are AKIEC, so the label encodes pathologist "
                "confidence rather than a distinct lesion type. It is kept anyway because "
                "P(Malignant)+P(Indeterminate) recovers the binary decision after "
                "inference, whereas merging at training time is irreversible."
            ),
        },
        "stretch_target": {
            "column": "dx",
            "n_classes": len(CLASS_CODES),
            "classes": CLASS_CODES,
            "map": DX_MAP,
            "class_names": CLASS_NAMES,
            "lesion_counts": {c: int(dx_counts.get(c, 0)) for c in CLASS_CODES},
            "malignant_classes": sorted(MALIGNANT),
            "decision": "keep all 11 classes; handle imbalance with weights + sampler",
            "rationale": (
                "Merging rare classes into 'other' is irreversible and yields a category "
                "no clinician can act on. Imbalance is handled in the loss weights and "
                "the WeightedRandomSampler instead."
            ),
            "low_support_classes": LOW_SUPPORT_CLASSES,
            "low_support_caveat": (
                "MAL_OTH has 9 lesions, so roughly 1 reaches the test split. Its per-class "
                "metric is reported for completeness but is not statistically meaningful."
            ),
        },
        "binary_collapse": BINARY_COLLAPSE,
        "class_to_diagnosis1": (
            lesions.groupby("dx")["diagnosis_1"]
            .agg(lambda s: sorted(set(s))).to_dict()
        ),
    }


def save_label_map(label_map: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2, sort_keys=False) + "\n")
    return path


def load_label_map(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def encode(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """String labels -> integers, raising on anything the map does not cover."""
    unknown = set(series.dropna().unique()) - set(mapping)
    if unknown:
        raise KeyError(f"labels not in the label map: {sorted(unknown)}")
    return series.map(mapping).astype("int64")


def class_weights(labels: pd.Series, mapping: dict[str, int],
                  scheme: str = "inverse") -> dict[str, float]:
    """Loss weights from the TRAIN split only (B8).

    ``inverse``: w_c = N / (K * n_c) - the sklearn 'balanced' convention. A class
    seen half as often gets twice the weight, and the weights average to 1 so the
    loss keeps its usual scale.
    """
    counts = labels.value_counts()
    n, k = len(labels), len(mapping)
    if scheme != "inverse":
        raise ValueError(f"unknown scheme {scheme!r}")
    return {c: float(n / (k * counts[c])) if counts.get(c, 0) else 0.0 for c in mapping}
