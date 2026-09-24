"""Property tests for the grouped lesion-level split (A3.2) and the label map (B3)."""

import numpy as np
import pandas as pd
import pytest

from milk10k import labels, splits


@pytest.fixture
def toy_lesions():
    """431 lesions across 5 classes, including two deliberately rare ones.

    The counts are deliberately NOT divisible by 5: with round numbers an
    unstratified GroupKFold splits them perfectly by accident and the
    stratification comparison below has nothing to detect.
    """
    rng = np.random.default_rng(0)
    counts = {"BCC": 203, "NV": 117, "MEL": 71, "DF": 23, "MAL_OTH": 17}
    rows = []
    i = 0
    for dx, n in counts.items():
        for _ in range(n):
            rows.append({
                "lesion_id": f"IL_{i:05d}",
                "derm_id": f"ISIC_D{i:05d}",
                "clinical_id": f"ISIC_C{i:05d}",
                "dx": dx,
                "diagnosis_1": "Malignant" if dx in {"BCC", "MEL", "MAL_OTH"} else "Benign",
                "age": float(rng.integers(20, 90)),
                "sex": rng.choice(["male", "female"]),
                "site": rng.choice(["head/neck", "torso", None]),
            })
            i += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def toy_images(toy_lesions):
    """The matching image table: two rows per lesion."""
    derm = toy_lesions.assign(isic_id=toy_lesions.derm_id, image_type="dermoscopic")
    clin = toy_lesions.assign(isic_id=toy_lesions.clinical_id, image_type="clinical")
    cols = ["isic_id", "lesion_id", "image_type", "dx", "diagnosis_1"]
    return pd.concat([derm[cols], clin[cols]], ignore_index=True)


# --------------------------------------------------------------------------
# A3.2 - the three required properties, over 10 seeds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_splits_are_disjoint(toy_lesions, seed):
    sp = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=seed)
    assert set(splits.check_disjoint(sp).values()) == {0}


@pytest.mark.parametrize("seed", range(10))
def test_split_sizes_within_one_percentage_point(toy_lesions, seed):
    sp = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=seed)
    check = splits.check_sizes(sp, 1 / 7, 1 / 7, tol_pp=1.0)
    assert check.within_tol.all(), check.to_string(index=False)


@pytest.mark.parametrize("seed", range(10))
def test_same_seed_gives_identical_output(toy_lesions, seed):
    a = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=seed)
    b = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=seed)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)


def test_different_seeds_give_different_splits(toy_lesions):
    a = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=0)
    b = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=1)
    assert a.test != b.test


def test_every_lesion_appears_exactly_once(toy_lesions):
    sp = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=0)
    assert len(sp) == len(toy_lesions)
    assert sorted(sp.train + sp.val + sp.test) == sorted(toy_lesions.lesion_id)


def test_rejects_impossible_sizes(toy_lesions):
    with pytest.raises(ValueError):
        splits.split_lesions(toy_lesions, 0.6, 0.6)
    with pytest.raises(ValueError):
        splits.split_lesions(toy_lesions, 0.0, 0.2)


def test_rejects_missing_columns(toy_lesions):
    with pytest.raises(KeyError):
        splits.split_lesions(toy_lesions.drop(columns=["dx"]))


# --------------------------------------------------------------------------
# The guarantee the whole module exists for
# --------------------------------------------------------------------------


def test_no_lesion_straddles_two_splits(toy_lesions, toy_images):
    sp = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=0)
    tagged = splits.images_for_split(toy_images, sp)
    assert tagged.groupby("lesion_id")["split"].nunique().eq(1).all()
    assert tagged.groupby("lesion_id").size().eq(2).all()


def test_only_the_ungrouped_strategy_leaks(toy_lesions, toy_images):
    """The part of A3.3 that is a hard guarantee: grouping eliminates leakage.

    Stratification *quality* is deliberately not asserted here - see
    ``test_stratification_helps_on_the_real_dataset``.
    """
    cmp = splits.compare_fold_strategies(toy_lesions, toy_images, n_splits=5, seed=0)
    by = cmp.set_index("strategy")

    assert by.loc["StratifiedGroupKFold (both)"].leaked_lesions == 0
    assert by.loc["GroupKFold (groups, no strat.)"].leaked_lesions == 0
    assert by.loc["StratifiedKFold (strat., no groups)"].leaked_lesions > 0


def test_stratification_helps_on_the_real_dataset():
    """The empirical half of A3.3, checked where the claim is actually made.

    That StratifiedGroupKFold balances classes better than GroupKFold is a
    property of *this* dataset (11 classes, counts from 9 to 2,522), not a
    mathematical guarantee: on a small, evenly-divisible synthetic set an
    unstratified split can match or beat it by luck. So the claim is tested
    against real MILK10k, and skipped when the data is not present.
    """
    import milk10k as mk

    if not mk.IMAGE_DIR.parent.joinpath("metadata.csv").is_file():
        pytest.skip("MILK10k metadata not available locally")

    images = mk.load_images_table()
    lesions = mk.build_lesion_table(images)
    cmp = splits.compare_fold_strategies(lesions, images, n_splits=5, seed=42)
    by = cmp.set_index("strategy")

    grouped = by.loc["StratifiedGroupKFold (both)"]
    unstratified = by.loc["GroupKFold (groups, no strat.)"]
    ungrouped = by.loc["StratifiedKFold (strat., no groups)"]

    assert grouped.leaked_lesions == 0
    assert ungrouped.leaked_lesions > 1000, "the naive image-level split must leak badly"
    assert grouped.max_class_spread_pp < unstratified.max_class_spread_pp


def test_nearest_achievable_respects_the_rare_class_cap():
    frac, n = splits.nearest_achievable(0.15, max_folds=9)
    assert n == 7 and abs(frac - 1 / 7) < 1e-12
    # a class with only 3 members caps the fold count at 3
    frac, n = splits.nearest_achievable(0.15, max_folds=3)
    assert n == 3


def test_save_and_reload_splits(toy_lesions, toy_images, tmp_path):
    sp = splits.split_lesions(toy_lesions, 1 / 7, 1 / 7, seed=0)
    paths = splits.save_splits(toy_images, sp, tmp_path)
    total = 0
    for name, p in paths.items():
        df = splits.load_split_csv(p)
        assert (df["split"] == name).all()
        total += len(df)
    assert total == len(toy_images)


# --------------------------------------------------------------------------
# B3 - label map and class weights
# --------------------------------------------------------------------------


def test_label_map_round_trips(toy_lesions, tmp_path):
    lm = labels.build_label_map(toy_lesions)
    path = labels.save_label_map(lm, tmp_path / "label_map.json")
    assert labels.load_label_map(path) == lm
    assert lm["primary_target"]["map"] == labels.DIAGNOSIS1_MAP


def test_encode_rejects_unknown_labels():
    s = pd.Series(["Benign", "Martian"])
    with pytest.raises(KeyError):
        labels.encode(s, labels.DIAGNOSIS1_MAP)


def test_class_weights_are_inverse_frequency():
    y = pd.Series(["Benign"] * 90 + ["Malignant"] * 10)
    mapping = {"Benign": 0, "Malignant": 1}
    w = labels.class_weights(y, mapping)
    # rarer class gets the larger weight, in proportion to the inverse count
    assert w["Malignant"] > w["Benign"]
    assert w["Malignant"] / w["Benign"] == pytest.approx(9.0)
