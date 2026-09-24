"""Measuring what a lesion-level leak actually *does* to a metric (A3.1).

Counting leaked lesions is cheap and unconvincing. This module trains the same
metadata-only model under a naive image-level split and under a grouped split,
and adds a "sibling oracle" that upper-bounds how much a model could gain by
recognising that two photos show the same lesion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#: Metadata-only inputs. Deliberately excludes every diagnosis_* column and
#: diagnosis_confirm_type, which leak the label outright (see B4).
NUMERIC_FEATURES = ["age_approx"]
CATEGORICAL_FEATURES = ["sex", "anatom_site_general", "image_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_metadata_model(seed: int = 0) -> Pipeline:
    """RandomForest on metadata, with imputation/encoding INSIDE the pipeline.

    Putting the preprocessing in the Pipeline is the point: ``fit`` then only
    ever sees training rows, so the imputer's medians and the encoder's
    categories cannot carry test information back into training.

    ``min_samples_leaf=1`` is deliberate - the forest is allowed to memorise, so
    that if the split leaks, the metric has every chance to show it.
    """
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1,
            random_state=seed, n_jobs=-1)),
    ])


def _one_split(images: pd.DataFrame, target: str, seed: int, grouped: bool,
               n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) positional indices for one fold."""
    y = images[target].to_numpy()
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(splitter.split(np.zeros(len(y)), y, images["lesion_id"].to_numpy()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))


def sibling_oracle(images: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                   target: str) -> tuple[np.ndarray, float]:
    """Predict each test image with its sibling's label, if the sibling is in train.

    This is the *potential* of the leak: it is what a model would score if it
    could do nothing except recognise "I have seen this exact lesion before".
    Test images whose sibling is not in train fall back to the train majority
    class. Returns (predictions, fraction of test images whose sibling is in train).
    """
    train = images.iloc[train_idx]
    test = images.iloc[test_idx]
    majority = train[target].value_counts().idxmax()

    # lesion -> label, restricted to lesions that appear in TRAIN
    train_lesion_label = train.set_index("lesion_id")[target]
    train_lesion_label = train_lesion_label[~train_lesion_label.index.duplicated()]

    sibling_in_train = test["lesion_id"].isin(train_lesion_label.index)
    preds = np.where(
        sibling_in_train,
        test["lesion_id"].map(train_lesion_label).fillna(majority),
        majority,
    )
    return preds, float(sibling_in_train.mean())


def run_leak_experiment(images: pd.DataFrame, target: str = "diagnosis_1",
                        seeds: range | list[int] = range(5),
                        n_splits: int = 5) -> pd.DataFrame:
    """A3.1 (a)-(c): RF and sibling-oracle balanced accuracy under both splits.

    One row per (split strategy, seed) with the random forest's balanced
    accuracy, the oracle's balanced accuracy, and the leak rate.
    """
    rows = []
    for grouped, name in [(False, "image-level (naive)"), (True, "grouped by lesion_id")]:
        for seed in seeds:
            tr, te = _one_split(images, target, seed, grouped, n_splits)
            X, y = images[FEATURES], images[target]

            model = build_metadata_model(seed)
            model.fit(X.iloc[tr], y.iloc[tr])
            rf_score = balanced_accuracy_score(y.iloc[te], model.predict(X.iloc[te]))

            oracle_pred, leak_rate = sibling_oracle(images, tr, te, target)
            oracle_score = balanced_accuracy_score(y.iloc[te], oracle_pred)

            # leaked lesions = lesions with images on both sides of the boundary
            leaked = len(set(images.iloc[tr]["lesion_id"]) & set(images.iloc[te]["lesion_id"]))

            rows.append({
                "split": name, "seed": seed,
                "rf_balanced_acc": rf_score,
                "oracle_balanced_acc": oracle_score,
                "pct_test_with_sibling_in_train": 100 * leak_rate,
                "leaked_lesions": leaked,
                "n_test": len(te),
            })
    return pd.DataFrame(rows)


def summarise_leak(results: pd.DataFrame) -> pd.DataFrame:
    """mean +- std across seeds, per split strategy."""
    agg = results.groupby("split").agg(
        rf_mean=("rf_balanced_acc", "mean"), rf_std=("rf_balanced_acc", "std"),
        oracle_mean=("oracle_balanced_acc", "mean"), oracle_std=("oracle_balanced_acc", "std"),
        sibling_pct=("pct_test_with_sibling_in_train", "mean"),
        leaked_lesions=("leaked_lesions", "mean"),
    ).round(4)
    agg["rf_balanced_acc"] = (agg.rf_mean.round(3).astype(str) + " +- "
                              + agg.rf_std.round(3).astype(str))
    agg["oracle_balanced_acc"] = (agg.oracle_mean.round(3).astype(str) + " +- "
                                  + agg.oracle_std.round(3).astype(str))
    return agg[["rf_balanced_acc", "oracle_balanced_acc", "sibling_pct", "leaked_lesions"]]
