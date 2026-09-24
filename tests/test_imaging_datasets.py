"""Tests for the pixel utilities (A2), the augmentation audit (A3.5) and the
PyTorch datasets (A3.6, B8)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from milk10k import imaging

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from milk10k import augment, datasets, labels  # noqa: E402


# --------------------------------------------------------------------------
# A2.1 - geometry and views
# --------------------------------------------------------------------------


@pytest.fixture
def rgb_image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))


def test_numpy_ops_match_pillow(rgb_image):
    check = imaging.verify_against_pillow(rgb_image)
    assert check.identical.all(), check.to_string(index=False)


def test_all_geometry_ops_return_views(rgb_image):
    arr = np.asarray(rgb_image)
    for fn in (imaging.flip_lr, imaging.flip_ud, imaging.transpose_hw,
               imaging.rotate90_ccw, imaging.rotate90_cw):
        assert np.shares_memory(fn(arr), arr), f"{fn.__name__} should be a view"


def test_writing_through_a_view_mutates_the_original():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    imaging.flip_lr(a)[0, 0] = 99
    assert a[0, 3] == 99, "a view write must reach the original"


def test_copy_does_not_mutate_the_original():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    imaging.flip_lr(a).copy()[0, 0] = 99
    assert a[0, 3] == 3


def test_memory_budget_arithmetic():
    b = imaging.memory_budget(1000, {"224": (224, 224, 3)})
    u8 = b[b.dtype == "uint8"].iloc[0]
    f32 = b[b.dtype == "float32"].iloc[0]
    assert u8.bytes_per_image == 224 * 224 * 3
    assert f32.bytes_per_image == 4 * u8.bytes_per_image
    assert f32.total_GiB == pytest.approx(4 * u8.total_GiB, rel=1e-3)


# --------------------------------------------------------------------------
# A2.2 - PSNR
# --------------------------------------------------------------------------


def test_psnr_of_identical_images_is_infinite():
    a = np.zeros((8, 8), dtype=np.uint8)
    assert imaging.psnr(a, a) == float("inf")


def test_psnr_matches_the_definition():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 255
    expected = 10 * np.log10(255**2 / ((255.0**2) / 16))
    assert imaging.psnr(a, b) == pytest.approx(expected)


def test_psnr_of_maximally_different_images_is_zero():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 255, dtype=np.uint8)
    assert imaging.psnr(a, b) == pytest.approx(0.0)


def test_psnr_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        imaging.psnr(np.zeros((4, 4)), np.zeros((5, 5)))


def test_psnr_is_monotonic_on_a_never_compressed_image():
    """The control from A2.2: without double compression, quality tracks PSNR."""
    x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 96))
    arr = np.stack([np.sin(8 * x) * 127 + 128, np.cos(6 * y) * 127 + 128,
                    (x + y) * 127], -1).astype(np.uint8)
    sweep, _ = imaging.quality_sweep(Image.fromarray(arr), qualities=(95, 75, 50, 25, 10))
    assert sweep.psnr_dB.is_monotonic_decreasing
    assert sweep.size_KB.is_monotonic_decreasing


def test_detect_jpeg_quality_recovers_the_setting(tmp_path):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    for q in (50, 75, 95):
        p = tmp_path / f"q{q}.jpg"
        img.save(p, quality=q)
        assert imaging.detect_jpeg_quality(p) == q


# --------------------------------------------------------------------------
# A3.5 - augmentation audit
# --------------------------------------------------------------------------


def test_circular_mean_wraps_around_zero():
    # 359 and 1 degrees average to 0, not to 180
    assert imaging is not None
    assert augment.circular_mean_degrees([359.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_circular_gap_takes_the_short_way_round():
    assert augment.circular_gap_degrees(350, 10) == pytest.approx(20.0)
    assert augment.circular_gap_degrees(10, 350) == pytest.approx(20.0)
    assert augment.circular_gap_degrees(0, 180) == pytest.approx(180.0)


def test_eval_transform_is_deterministic(rgb_image):
    assert augment.assert_eval_deterministic(rgb_image)


def test_train_transform_is_random(rgb_image):
    tf = augment.train_transform()
    assert not torch.equal(tf(rgb_image), tf(rgb_image))


def test_transforms_produce_the_configured_shape(rgb_image):
    for tf in (augment.train_transform(64), augment.eval_transform(64)):
        assert tuple(tf(rgb_image).shape) == (3, 64, 64)


# --------------------------------------------------------------------------
# A3.6 / B8 - datasets
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path):
    """4 lesions x 2 views on disk, plus the matching tables."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    rows, lesion_rows = [], []
    for i in range(4):
        ids = {}
        for view, prefix in (("derm", "D"), ("clinical", "C")):
            isic = f"ISIC_{prefix}{i:04d}"
            Image.fromarray(
                rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            ).save(img_dir / f"{isic}.jpg", quality=90)
            ids[view] = isic
            rows.append({"isic_id": isic, "lesion_id": f"IL_{i}",
                         "image_type": view,
                         "diagnosis_1": "Malignant" if i % 2 else "Benign"})
        lesion_rows.append({"lesion_id": f"IL_{i}", "derm_id": ids["derm"],
                            "clinical_id": ids["clinical"],
                            "diagnosis_1": "Malignant" if i % 2 else "Benign"})
    return img_dir, pd.DataFrame(lesion_rows), pd.DataFrame(rows)


def test_lesion_dataset_returns_both_views(tiny_dataset):
    img_dir, lesions, _ = tiny_dataset
    ds = datasets.LesionDataset(lesions, img_dir=img_dir, label_col="diagnosis_1",
                                transform=augment.eval_transform(32),
                                label_map=labels.DIAGNOSIS1_MAP)
    item = ds[0]
    assert set(item) == {"derm", "clinical", "label", "lesion_id"}
    assert tuple(item["derm"].shape) == (3, 32, 32)
    assert item["lesion_id"] == "IL_0"


def test_lesion_dataset_batches_with_dataloader(tiny_dataset):
    img_dir, lesions, _ = tiny_dataset
    ds = datasets.LesionDataset(lesions, img_dir=img_dir, label_col="diagnosis_1",
                                transform=augment.eval_transform(32),
                                label_map=labels.DIAGNOSIS1_MAP)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert tuple(batch["derm"].shape) == (4, 3, 32, 32)
    assert tuple(batch["clinical"].shape) == (4, 3, 32, 32)
    lookup = lesions.set_index("lesion_id").diagnosis_1
    assert all(labels.DIAGNOSIS1_MAP[lookup[i]] == int(l)
               for i, l in zip(batch["lesion_id"], batch["label"]))


def test_one_epoch_sees_every_label(tiny_dataset):
    img_dir, lesions, _ = tiny_dataset
    ds = datasets.LesionDataset(lesions, img_dir=img_dir, label_col="diagnosis_1",
                                transform=augment.eval_transform(32),
                                label_map=labels.DIAGNOSIS1_MAP)
    seen = {}
    for b in DataLoader(ds, batch_size=2):
        for y in b["label"].tolist():
            seen[y] = seen.get(y, 0) + 1
    expected = {labels.DIAGNOSIS1_MAP[k]: v
                for k, v in lesions.diagnosis_1.value_counts().items()}
    assert seen == expected


def test_transform_applied_independently_per_view(tiny_dataset):
    img_dir, lesions, _ = tiny_dataset
    ds = datasets.LesionDataset(lesions, img_dir=img_dir, label_col="diagnosis_1",
                                transform=augment.train_transform(32),
                                label_map=labels.DIAGNOSIS1_MAP)
    item = ds[0]
    assert not torch.equal(item["derm"], item["clinical"])


def test_missing_image_fails_loudly(tiny_dataset):
    img_dir, lesions, _ = tiny_dataset
    broken = lesions.copy()
    broken.loc[0, "derm_id"] = "ISIC_NOPE"
    ds = datasets.LesionDataset(broken, img_dir=img_dir, label_col="diagnosis_1",
                                label_map=labels.DIAGNOSIS1_MAP)
    with pytest.raises(datasets.MissingImageError):
        _ = ds[0]


def test_lesion_dataset_rejects_unknown_view(tiny_dataset):
    img_dir, lesions, _ = tiny_dataset
    with pytest.raises(ValueError):
        datasets.LesionDataset(lesions, img_dir=img_dir, views=("xray",))


def test_image_dataset_returns_id(tiny_dataset):
    img_dir, _, images = tiny_dataset
    ds = datasets.MILK10kImageDataset(images, img_dir=img_dir, label_col="diagnosis_1",
                                      label_map=labels.DIAGNOSIS1_MAP,
                                      transform=augment.eval_transform(32))
    tensor, label, isic_id = ds[0]
    assert tuple(tensor.shape) == (3, 32, 32)
    assert isinstance(label, int) and isic_id.startswith("ISIC_")


def test_weighted_sampler_balances_an_imbalanced_split(tiny_dataset):
    img_dir, _, images = tiny_dataset
    skewed = pd.concat([images[images.diagnosis_1 == "Benign"],
                        images[images.diagnosis_1 == "Malignant"].head(1)])
    ds = datasets.MILK10kImageDataset(skewed, img_dir=img_dir, label_col="diagnosis_1",
                                      label_map=labels.DIAGNOSIS1_MAP,
                                      transform=augment.eval_transform(32))
    sampler = datasets.make_weighted_sampler(ds)
    drawn = np.array([ds.table.iloc[i].diagnosis_1 for i in list(sampler)])
    # with inverse-frequency weights the rare class should be heavily up-sampled
    # relative to its 1-in-5 share; allow a wide band because n is tiny
    assert 0.15 < (drawn == "Malignant").mean() < 0.9


def test_aggregate_predictions_averages_two_views():
    table = pd.DataFrame({"isic_id": list("abcd"),
                          "lesion_id": ["L1", "L1", "L2", "L2"]})
    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.3, 0.7]])
    out = datasets.aggregate_predictions(probs, table, classes=["Benign", "Malignant"])
    assert list(out.lesion_id) == ["L1", "L2"]
    assert (out.n_views == 2).all()
    assert out.loc[0, "Benign"] == pytest.approx(0.7)
    assert list(out.pred_label) == ["Benign", "Malignant"]


def test_aggregate_predictions_validates_shapes():
    table = pd.DataFrame({"isic_id": ["a"], "lesion_id": ["L1"]})
    with pytest.raises(ValueError):
        datasets.aggregate_predictions(np.zeros((2, 2)), table)
    with pytest.raises(ValueError):
        datasets.aggregate_predictions(np.zeros(2), table)
