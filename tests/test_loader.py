import numpy as np
import pytest

from milk10k.loader import MILK10kLoader
from milk10k.preprocess import PreprocessConfig


def test_loader_filters_to_available_files(synthetic_dataset):
    img_dir, df = synthetic_dataset
    loader = MILK10kLoader(df, PreprocessConfig(size=(16, 16)), batch_size=4, shuffle=False)
    # ISIC_MISSING has no file -> dropped at construction; ISIC_BAD exists (corrupt) -> kept, skipped at iteration
    assert loader.n_images == 7 and loader.n_unavailable == 1
    assert "ISIC_MISSING" not in set(loader.df.isic_id)


def test_loader_yields_batches_with_labels(synthetic_dataset):
    _, df = synthetic_dataset
    loader = MILK10kLoader(df, PreprocessConfig(size=(16, 16)), batch_size=4, shuffle=False)
    assert len(loader) == 2                      # 7 images / 4 -> 2 batches
    batches = list(loader)
    assert len(batches) == 2
    b0 = batches[0]
    assert b0.images.shape == (4, 16, 16, 3) and b0.labels.shape == (4,)
    assert b0.labels.dtype == np.int64
    assert b0.label_names == list(df.label[:4])
    # corrupt file is in the 2nd batch -> skipped, so images/labels stay aligned
    b1 = batches[1]
    assert len(b1) == 2 and b1.images.shape[0] == len(b1.labels) == 2
    assert [i for i, _ in b1.skipped] == ["ISIC_BAD"]


def test_labels_map_to_classes_in_canonical_order(synthetic_dataset):
    _, df = synthetic_dataset
    loader = MILK10kLoader(df, PreprocessConfig(size=(8, 8)), batch_size=8, shuffle=False)
    assert loader.classes == ["BCC", "MEL", "NV"]        # CLASS_CODES order
    b = next(iter(loader))
    for idx, isic in zip(b.labels, b.ids):
        assert loader.classes[idx] == df.set_index("isic_id").label[isic]


def test_shuffle_is_reproducible_with_seed(synthetic_dataset):
    _, df = synthetic_dataset
    cfg = PreprocessConfig(size=(8, 8))
    a = [b.ids for b in MILK10kLoader(df, cfg, batch_size=3, shuffle=True, seed=7)]
    b = [b.ids for b in MILK10kLoader(df, cfg, batch_size=3, shuffle=True, seed=7)]
    c = [b.ids for b in MILK10kLoader(df, cfg, batch_size=3, shuffle=True, seed=8)]
    assert a == b and a != c


def test_drop_last_and_class_counts(synthetic_dataset):
    _, df = synthetic_dataset
    loader = MILK10kLoader(df, PreprocessConfig(size=(8, 8)), batch_size=4, drop_last=True)
    assert len(loader) == 1
    counts = loader.class_counts()
    assert counts["NV"] == 3 and counts["BCC"] == 2 and counts["MEL"] == 2


def test_loader_rejects_bad_inputs(synthetic_dataset):
    _, df = synthetic_dataset
    with pytest.raises(ValueError):
        MILK10kLoader(df, batch_size=0)
    with pytest.raises(ValueError):
        MILK10kLoader(df.drop(columns=["label"]))
    with pytest.raises(ValueError):
        MILK10kLoader(df[df.isic_id == "ISIC_MISSING"])   # nothing on disk


def test_real_loader_smoke(real_images):
    import milk10k
    df = milk10k.load_images_table().head(40)
    loader = MILK10kLoader(df, PreprocessConfig(size=(64, 64)), batch_size=16, shuffle=False)
    b = next(iter(loader))
    assert b.images.shape == (16, 64, 64, 3) and b.skipped == []
    assert set(b.label_names) <= set(milk10k.CLASS_CODES)


def test_strict_loader_fails_loudly_on_missing_files(synthetic_dataset):
    """B1: the project pipeline must never silently skip a missing image."""
    import pytest

    _, df = synthetic_dataset          # contains one row with no file on disk
    with pytest.raises(FileNotFoundError, match="no image on disk"):
        MILK10kLoader(df, PreprocessConfig(size=(16, 16)), strict=True)


def test_non_strict_loader_still_skips(synthetic_dataset):
    _, df = synthetic_dataset
    loader = MILK10kLoader(df, PreprocessConfig(size=(16, 16)), strict=False)
    assert loader.n_unavailable == 1
