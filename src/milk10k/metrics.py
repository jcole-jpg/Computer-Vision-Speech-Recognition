"""Baselines and cost-aware evaluation (A3.4).

Accuracy treats every mistake as equal. In skin-cancer triage it plainly is not:
sending a benign lesion for an unnecessary biopsy wastes a clinic slot, missing
a melanoma can cost a life. This module scores predictors under both views so
the disagreement between them is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

#: Rows = truth, columns = prediction. Correct predictions cost 0; missing a
#: Malignant costs 50; an unnecessary work-up of a Benign costs 1; anything
#: involving Indeterminate costs 5. These are the homework's suggested numbers -
#: a real deployment would have a clinician set them.
DEFAULT_CLASSES = ("Benign", "Indeterminate", "Malignant")
DEFAULT_COST = pd.DataFrame(
    [
        # pred:  Benign  Indeterminate  Malignant
        [0,  1,  1],   # truth Benign        -> unnecessary work-up
        [5,  0,  5],   # truth Indeterminate
        [50, 5,  0],   # truth Malignant     -> a miss is the expensive error
    ],
    index=list(DEFAULT_CLASSES),
    columns=list(DEFAULT_CLASSES),
)


def score_predictions(y_true, y_pred, labels=DEFAULT_CLASSES) -> dict[str, float]:
    """accuracy / balanced accuracy / macro-F1 for one set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             labels=list(labels), zero_division=0),
    }


def expected_cost(y_true, y_pred, cost: pd.DataFrame = DEFAULT_COST) -> float:
    """Mean cost per lesion under the cost matrix."""
    y_true = pd.Series(list(y_true), dtype=object)
    y_pred = pd.Series(list(y_pred), dtype=object)
    return float(np.mean([cost.loc[t, p] for t, p in zip(y_true, y_pred)]))


def evaluate_dummies(y_train, y_test, cost: pd.DataFrame = DEFAULT_COST,
                     classes=DEFAULT_CLASSES, seed: int = 0
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Score the dummy baselines plus the two constant predictors (A3.4 a & b).

    Returns (table, confusion matrix of the stratified-random dummy).
    """
    X_train = np.zeros((len(y_train), 1))
    X_test = np.zeros((len(y_test), 1))

    strategies = {
        "always-Malignant": DummyClassifier(strategy="constant", constant="Malignant"),
        "always-Benign": DummyClassifier(strategy="constant", constant="Benign"),
        "stratified-random": DummyClassifier(strategy="stratified", random_state=seed),
        "uniform-random": DummyClassifier(strategy="uniform", random_state=seed),
    }

    rows, strat_cm = [], None
    for name, clf in strategies.items():
        clf.fit(X_train, list(y_train))
        pred = clf.predict(X_test)
        row = {"predictor": name, **score_predictions(y_test, pred, classes)}
        row["expected_cost_per_lesion"] = expected_cost(y_test, pred, cost)
        rows.append(row)
        if name == "stratified-random":
            strat_cm = confusion_matrix(y_test, pred, labels=list(classes))

    table = pd.DataFrame(rows).round(4)
    table["rank_by_accuracy"] = table["accuracy"].rank(ascending=False).astype(int)
    table["rank_by_cost"] = table["expected_cost_per_lesion"].rank(ascending=True).astype(int)
    return table, strat_cm


def constant_prediction_costs(y_true, cost: pd.DataFrame = DEFAULT_COST,
                              classes=DEFAULT_CLASSES) -> pd.DataFrame:
    """Expected cost of predicting one constant class for everything (A3.4 c)."""
    rows = []
    for c in classes:
        pred = [c] * len(y_true)
        rows.append({
            "constant_prediction": c,
            "accuracy": accuracy_score(y_true, pred),
            "expected_cost_per_lesion": expected_cost(y_true, pred, cost),
        })
    return pd.DataFrame(rows).round(4).sort_values("expected_cost_per_lesion")
