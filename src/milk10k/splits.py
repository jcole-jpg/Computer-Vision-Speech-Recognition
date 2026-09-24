"""Leak-free train/val/test splitting, grouped by lesion (A3.2, A3.3, B5).

The one rule this module exists to enforce: **a lesion never straddles two
splits.** MILK10k gives every lesion two photographs of the same skin; if one
lands in train and the other in test, the test score measures memorisation of
that lesion, not generalisation to a new patient.

Splitting therefore happens on the *lesion* table (one row per lesion) and
images are assigned afterwards by looking up their lesion's split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from .config import CONFIG, SEED


@dataclass(frozen=True)
class LesionSplit:
    """Three disjoint lists of ``lesion_id``."""

    train: list[str]
    val: list[str]
    test: list[str]

    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    @property
    def fractions(self) -> dict[str, float]:
        n = len(self)
        return {k: v / n for k, v in self.sizes.items()}

    def as_series(self) -> pd.Series:
        """lesion_id -> split name, handy for joining onto the image table."""
        pairs = (
            [(i, "train") for i in self.train]
            + [(i, "val") for i in self.val]
            + [(i, "test") for i in self.test]
        )
        ids, names = zip(*pairs)
        return pd.Series(list(names), index=list(ids), name="split")


def nearest_achievable(fraction: float, max_folds: int) -> tuple[float, int]:
    """The hold-out fraction a single ``StratifiedGroupKFold`` fold can actually hit.

    A fold-based splitter holds out ``1/n_splits`` of the data, so achievable
    fractions are quantised to 1/n. ``max_folds`` should be the size of the
    rarest class: ask for more folds than that and some fold necessarily
    contains none of it, which is how a rare class silently vanishes from the
    validation or test set.

    Returns ``(achievable_fraction, n_splits)``.
    """
    n = max(2, min(max_folds, round(1 / fraction)))
    return 1 / n, n


def _carve_one_fold(y: np.ndarray, groups: np.ndarray, fraction: float,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ~``fraction`` of rows as one StratifiedGroupKFold fold.

    ``n_splits = round(1/fraction)`` folds each hold ~``1/n_splits`` of the data,
    so the realised fraction is the achievable one nearest the request (e.g. a
    requested 0.15 becomes 1/7 = 0.143). The caller checks the tolerance.
    """
    n_splits = max(2, round(1 / fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return rest_idx, held_idx


def split_lesions(lesions: pd.DataFrame, val_size: float = CONFIG.val_size,
                  test_size: float = CONFIG.test_size, seed: int = SEED,
                  label_col: str = "dx", group_col: str = "lesion_id") -> LesionSplit:
    """Split lesions into train/val/test, stratified on ``label_col`` (A3.2).

    Two nested StratifiedGroupKFold carves: first the test set, then the
    validation set out of what remains. ``groups`` is the lesion id, so the
    grouping guarantee still holds if this is ever pointed at an image-level
    table (where a group really does span several rows).

    Stratification uses the 11-class ``dx`` by default: it is the finest label,
    so balancing it also balances the coarser ``diagnosis_1`` (B5).

    Same ``seed`` always returns the same three lists.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be fractions in (0, 1)")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must leave room for a train split")
    for col in (label_col, group_col):
        if col not in lesions.columns:
            raise KeyError(f"lesion table has no column {col!r}")

    y = lesions[label_col].to_numpy()
    groups = lesions[group_col].to_numpy()

    # 1) carve the test set out of everything
    rest_idx, test_idx = _carve_one_fold(y, groups, test_size, seed)

    # 2) carve validation out of the remainder; the requested val_size is a
    #    fraction of the WHOLE set, so rescale it to the remainder's size.
    val_fraction_of_rest = val_size / (1.0 - len(test_idx) / len(lesions))
    sub_rest_idx, sub_val_idx = _carve_one_fold(
        y[rest_idx], groups[rest_idx], val_fraction_of_rest, seed
    )

    ids = lesions[group_col].to_numpy()
    return LesionSplit(
        train=ids[rest_idx[sub_rest_idx]].tolist(),
        val=ids[rest_idx[sub_val_idx]].tolist(),
        test=ids[test_idx].tolist(),
    )


# --------------------------------------------------------------------------
# Verification (A3.2 property test, B5 printed checks)
# --------------------------------------------------------------------------


def check_disjoint(split: LesionSplit) -> dict[str, int]:
    """Size of the overlap between each pair of splits - all must be 0."""
    tr, va, te = set(split.train), set(split.val), set(split.test)
    return {
        "train&val": len(tr & va),
        "train&test": len(tr & te),
        "val&test": len(va & te),
    }


def check_sizes(split: LesionSplit, val_size: float, test_size: float,
                tol_pp: float = 1.0) -> pd.DataFrame:
    """Realised vs requested split fractions, with the deviation in percentage points."""
    want = {"train": 1 - val_size - test_size, "val": val_size, "test": test_size}
    got = split.fractions
    rows = [
        {"split": k, "requested_pct": 100 * want[k], "actual_pct": 100 * got[k],
         "deviation_pp": abs(100 * got[k] - 100 * want[k]),
         "within_tol": abs(100 * got[k] - 100 * want[k]) <= tol_pp}
        for k in ("train", "val", "test")
    ]
    return pd.DataFrame(rows)


def class_proportions(lesions: pd.DataFrame, split: LesionSplit,
                      label_col: str = "dx") -> pd.DataFrame:
    """Per-split class proportions (%) plus the global column and max deviation (B5)."""
    mapping = split.as_series()
    df = lesions.assign(split=lesions["lesion_id"].map(mapping))
    tab = (
        df.groupby(["split", label_col]).size()
        .unstack(label_col, fill_value=0)
        .pipe(lambda t: 100 * t.div(t.sum(axis=1), axis=0))
        .T
    )
    tab["global"] = 100 * lesions[label_col].value_counts(normalize=True)
    tab["max_dev_pp"] = (
        tab[[c for c in ("train", "val", "test") if c in tab.columns]]
        .sub(tab["global"], axis=0).abs().max(axis=1)
    )
    return tab.sort_values("global", ascending=False).round(2)


def rare_class_coverage(lesions: pd.DataFrame, seeds: range | list[int],
                        rare: tuple[str, ...] = ("MAL_OTH", "DF", "INF", "VASC", "BEN_OTH"),
                        label_col: str = "dx", **kwargs) -> pd.DataFrame:
    """Over N seeds, how often is each rare class present in val AND in test? (A3.2)"""
    counts = {c: 0 for c in rare}
    for seed in seeds:
        sp = split_lesions(lesions, seed=seed, label_col=label_col, **kwargs)
        in_val = set(lesions.loc[lesions["lesion_id"].isin(sp.val), label_col])
        in_test = set(lesions.loc[lesions["lesion_id"].isin(sp.test), label_col])
        for c in rare:
            if c in in_val and c in in_test:
                counts[c] += 1
    n = len(list(seeds))
    return pd.DataFrame(
        [{"class": c,
          "n_lesions": int((lesions[label_col] == c).sum()),
          "seeds_in_val_and_test": counts[c],
          "n_seeds": n,
          "pct": round(100 * counts[c] / n, 1)}
         for c in rare]
    ).sort_values("n_lesions")


# --------------------------------------------------------------------------
# A3.3 - why StratifiedGroupKFold?
# --------------------------------------------------------------------------


def compare_fold_strategies(lesions: pd.DataFrame, images: pd.DataFrame,
                            n_splits: int = 5, seed: int = SEED,
                            label_col: str = "dx",
                            rare_class: str = "MAL_OTH") -> pd.DataFrame:
    """Compare GroupKFold / StratifiedKFold / StratifiedGroupKFold (A3.3).

    Reports, per strategy: leaked lesions summed over folds (a lesion with images
    on both sides of a fold boundary), the largest max-min spread of any class
    proportion across the validation folds, and how many validation folds miss
    ``rare_class`` entirely.

    StratifiedKFold is deliberately run on the IMAGE table - that is exactly the
    naive setup that leaks, and the point of the comparison is to price it.
    """
    lesion_of_image = images.set_index("isic_id")["lesion_id"]
    label_of_lesion = lesions.set_index("lesion_id")[label_col]
    rows = []

    strategies = {
        "GroupKFold (groups, no strat.)": ("lesion", GroupKFold(n_splits=n_splits), True),
        "StratifiedKFold (strat., no groups)": (
            "image", StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), False),
        "StratifiedGroupKFold (both)": (
            "lesion", StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), True),
    }

    for name, (level, splitter, uses_groups) in strategies.items():
        if level == "lesion":
            y = lesions[label_col].to_numpy()
            groups = lesions["lesion_id"].to_numpy()
            unit_ids = lesions["lesion_id"].to_numpy()
        else:
            y = images["isic_id"].map(lesion_of_image).map(label_of_lesion).to_numpy()
            groups = images["isic_id"].map(lesion_of_image).to_numpy()
            unit_ids = images["isic_id"].to_numpy()

        leaked, class_props, absent = 0, [], 0
        args = (np.zeros(len(y)), y, groups) if uses_groups else (np.zeros(len(y)), y)
        for train_idx, val_idx in splitter.split(*args):
            if level == "image":
                tr_les = set(lesion_of_image[unit_ids[train_idx]])
                va_les = set(lesion_of_image[unit_ids[val_idx]])
            else:
                tr_les, va_les = set(unit_ids[train_idx]), set(unit_ids[val_idx])
            leaked += len(tr_les & va_les)

            props = pd.Series(y[val_idx]).value_counts(normalize=True) * 100
            class_props.append(props)
            if props.get(rare_class, 0) == 0:
                absent += 1

        spread = pd.DataFrame(class_props).fillna(0.0)
        rows.append({
            "strategy": name,
            "level": level,
            "leaked_lesions": leaked,
            "max_class_spread_pp": round(float((spread.max() - spread.min()).max()), 2),
            f"folds_without_{rare_class}": absent,
            "n_folds": n_splits,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Persistence (B5)
# --------------------------------------------------------------------------


def images_for_split(images: pd.DataFrame, split: LesionSplit) -> pd.DataFrame:
    """Attach a ``split`` column to the image table by looking up each lesion."""
    return images.assign(split=images["lesion_id"].map(split.as_series()))


def save_splits(images: pd.DataFrame, split: LesionSplit, out_dir: Path,
                columns: tuple[str, ...] = ("isic_id", "lesion_id", "image_type",
                                            "diagnosis_1", "label", "split")) -> dict[str, Path]:
    """Write train/val/test CSVs (one row per image) and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tagged = images_for_split(images, split)
    keep = [c for c in columns if c in tagged.columns]
    paths = {}
    for name in ("train", "val", "test"):
        part = tagged[tagged["split"] == name][keep].sort_values("isic_id")
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        paths[name] = path
    return paths


def load_split_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
