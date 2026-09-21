import numpy as np
import pandas as pd
import pytest

from milk10k import stats, color


def test_cramers_v_extremes():
    perfect = np.diag([50, 50, 50])                 # category fully determines class
    independent = np.full((3, 3), 40)               # identical rows
    assert stats.cramers_v(perfect) > 0.95
    assert stats.cramers_v(independent) == pytest.approx(0.0, abs=1e-9)


def test_epsilon_squared_bounds():
    assert stats.epsilon_squared(h=0.0, n=100, k=3) == 0.0
    assert 0.0 < stats.epsilon_squared(h=50.0, n=100, k=3) < 1.0


def _toy_table():
    rng = np.random.default_rng(0)
    n = 300
    label = rng.choice(["BCC", "NV", "MEL"], size=n)
    age = np.where(label == "NV", 30, 70) + rng.normal(0, 5, n)      # strongly associated
    sex = rng.choice(["male", "female"], size=n)                     # independent
    return pd.DataFrame({"isic_id": [f"I{i}" for i in range(n)], "lesion_id": [f"L{i}" for i in range(n)],
                         "label": label, "age_approx": age, "sex": sex,
                         "attribution": "x", "image_type": rng.choice(["a", "b"], size=n),
                         "diagnosis_1": label, "path": "nope"})


def test_numeric_and_categorical_tests_pick_up_signal():
    df = _toy_table()
    num = stats.numeric_vs_target(df, "age_approx")
    cat = stats.categorical_vs_target(df, "sex")
    assert num["p_value"] < 1e-10 and num["effect_size"] > 0.5
    assert cat["p_value"] > 0.001 and cat["effect_size"] < 0.2
    assert list(cat["counts"].columns) == ["BCC", "MEL", "NV"]        # CLASS_CODES order


def test_describe_fields_types_and_leakage_flag():
    desc = stats.describe_fields(_toy_table()).set_index("field")
    assert desc.loc["age_approx", "kind"] == "numeric"
    assert desc.loc["sex", "kind"] == "categorical"
    assert desc.loc["attribution", "kind"] == "constant"
    assert desc.loc["diagnosis_1", "note"].startswith("LEAKAGE")
    assert "isic_id" not in desc.index and "label" not in desc.index


def test_association_table_excludes_leakage_and_constants():
    df = _toy_table()
    summary, results = stats.association_table(df, df)
    assert "diagnosis_1" not in set(summary.field) and "attribution" not in set(summary.field)
    assert summary.iloc[0].field == "age_approx"                      # ranked first
    assert "effect_r" in summary.columns


def test_sample_per_class_is_balanced_and_seeded():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"isic_id": [f"I{i}" for i in range(500)],
                       "label": rng.choice(["BCC", "NV", "DF"], size=500, p=[0.8, 0.18, 0.02]),
                       "image_type": rng.choice(["clinical", "dermoscopic"], size=500)})
    a = color.sample_per_class(df, 10, seed=3)
    b = color.sample_per_class(df, 10, seed=3)
    counts = a.groupby(["label", "image_type"]).size()
    assert counts.max() == 10 and (counts <= 10).all()
    assert a.isic_id.tolist() == b.isic_id.tolist()


def test_image_color_stats_on_synthetic(tmp_path):
    from PIL import Image
    arr = np.zeros((60, 80, 3), dtype=np.uint8); arr[..., 0] = 200; arr[..., 2] = 50
    p = tmp_path / "ISIC_X.jpg"; Image.fromarray(arr).save(p, quality=100)
    st = color.image_color_stats(p)
    assert st.isic_id == "ISIC_X"
    assert st.r_mean > st.b_mean > st.g_mean - 1
    assert st.gray_hist.shape == (color.N_BINS,) and st.rgb_hist.shape == (3, color.N_BINS)
    assert np.isclose(st.gray_hist.sum() * 4, 1.0, atol=0.02)       # density over 64 bins of width 4
