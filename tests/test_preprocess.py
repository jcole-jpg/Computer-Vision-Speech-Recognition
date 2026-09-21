import numpy as np
import pytest

from milk10k.preprocess import PreprocessConfig, preprocess_image, preprocess_batch


def test_default_config_rgb_minmax(synthetic_dataset):
    _, df = synthetic_dataset
    arr, info = preprocess_image(df.path[0])
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= info.value_min and info.value_max <= 1.0
    assert info.input_shape == (480, 640, 3)
    assert info.output_shape == (224, 224, 3)
    assert any(s.startswith("resize[crop]") for s in info.steps)


@pytest.mark.parametrize("size", [(64, 64), (128, 96), (50, 200)])
@pytest.mark.parametrize("resize_mode", ["crop", "stretch"])
def test_resize_modes_hit_target_shape(synthetic_dataset, size, resize_mode):
    _, df = synthetic_dataset
    cfg = PreprocessConfig(size=size, resize_mode=resize_mode)
    for p in df.path[:6]:                      # portrait, landscape, square inputs
        arr, info = preprocess_image(p, cfg)
        assert arr.shape == (*size, 3)
        assert info.output_shape == (*size, 3)


def test_grayscale_has_one_channel(synthetic_dataset):
    _, df = synthetic_dataset
    arr, info = preprocess_image(df.path[0], PreprocessConfig(size=(32, 32), color_mode="gray"))
    assert arr.shape == (32, 32, 1)
    assert "rgb->gray(luma)" in info.steps


def test_zscore_per_image_is_standardised(synthetic_dataset):
    _, df = synthetic_dataset
    arr, info = preprocess_image(df.path[0], PreprocessConfig(size=(64, 64), normalize="zscore"))
    assert abs(arr.mean()) < 1e-3
    assert abs(arr.std() - 1.0) < 1e-2
    assert "zscore(per-image mean/std)" in info.steps


def test_zscore_with_dataset_stats(synthetic_dataset):
    _, df = synthetic_dataset
    cfg = PreprocessConfig(size=(64, 64), normalize="zscore", mean=(127.5,) * 3, std=(50.0,) * 3)
    arr, info = preprocess_image(df.path[0], cfg)
    raw, _ = preprocess_image(df.path[0], PreprocessConfig(size=(64, 64), normalize=None))
    np.testing.assert_allclose(arr, (raw - 127.5) / 50.0, rtol=1e-5)
    assert "zscore(dataset mean/std)" in info.steps


def test_no_normalise_keeps_0_255(synthetic_dataset):
    _, df = synthetic_dataset
    arr, info = preprocess_image(df.path[0], PreprocessConfig(size=(32, 32), normalize=None))
    assert info.value_max > 1.0 and info.value_max <= 255.0


def test_accepts_numpy_array_input():
    rng = np.random.default_rng(1)
    src = rng.integers(0, 256, size=(100, 80, 3), dtype=np.uint8)
    arr, info = preprocess_image(src, PreprocessConfig(size=(40, 40)))
    assert arr.shape == (40, 40, 3) and info.source == "<array>"
    gray_src = rng.integers(0, 256, size=(100, 80), dtype=np.uint8)
    arr2, _ = preprocess_image(gray_src, PreprocessConfig(size=(40, 40)))
    assert arr2.shape == (40, 40, 3)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_image(tmp_path / "nope.jpg")


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PreprocessConfig(color_mode="cmyk")
    with pytest.raises(ValueError):
        PreprocessConfig(normalize="l2")
    with pytest.raises(ValueError):
        PreprocessConfig(mean=(1, 1, 1))          # std missing


def test_batch_skips_bad_and_missing_files(synthetic_dataset):
    _, df = synthetic_dataset
    res = preprocess_batch(df, PreprocessConfig(size=(32, 32)))
    assert res.images.shape == (6, 32, 32, 3)
    assert res.ids == list(df.isic_id[:6])
    skipped_ids = {i for i, _ in res.skipped}
    assert skipped_ids == {"ISIC_BAD", "ISIC_MISSING"}
    reasons = dict(res.skipped)
    assert "FileNotFoundError" in reasons["ISIC_MISSING"]
    assert "UnidentifiedImageError" in reasons["ISIC_BAD"]


def test_batch_accepts_list_and_dict(synthetic_dataset):
    _, df = synthetic_dataset
    cfg = PreprocessConfig(size=(16, 16))
    from_list = preprocess_batch(list(df.path[:3]), cfg)
    from_dict = preprocess_batch(dict(zip(df.isic_id[:3], df.path[:3])), cfg)
    assert from_list.images.shape == from_dict.images.shape == (3, 16, 16, 3)
    assert from_dict.ids == list(df.isic_id[:3])


def test_batch_all_failed_returns_empty_stack(tmp_path):
    res = preprocess_batch([tmp_path / "a.jpg", tmp_path / "b.jpg"], PreprocessConfig(size=(8, 8)))
    assert res.images.shape == (0, 8, 8, 3) and res.n_skipped == 2


def test_real_milk10k_images(real_images):
    res = preprocess_batch(real_images, PreprocessConfig(size=(128, 128)))
    assert res.images.shape == (len(real_images), 128, 128, 3)
    assert res.skipped == []
    assert 0.0 <= res.images.min() and res.images.max() <= 1.0
