import numpy as np

from put_pdn.data import TripletDataset, gaussian_downsample
from put_pdn.metrics import fusion_metrics


def test_patch_alignment_and_degradation():
    target = np.random.default_rng(1).random((8, 32, 32), dtype=np.float32)
    pan = target.mean(axis=0, keepdims=True)
    lr_hsi = gaussian_downsample(target, ratio=4)
    dataset = TripletDataset(
        pan,
        target,
        lr_hsi,
        ratio=4,
        patch_size=16,
        stride=8,
        augment=False,
    )
    sample = dataset[0]
    assert sample["pan"].shape == (1, 16, 16)
    assert sample["target"].shape == (8, 16, 16)
    assert sample["lr_hsi"].shape == (8, 4, 4)


def test_identical_metrics():
    image = np.random.default_rng(2).random((8, 16, 16))
    metrics = fusion_metrics(image, image, ratio=4, backend="standard")
    assert np.isinf(metrics["psnr"])
    assert np.isclose(metrics["ssim"], 1.0)
    assert np.isclose(metrics["sam"], 0.0, atol=1e-6)
    assert np.isclose(metrics["ergas"], 0.0)
    assert np.isclose(metrics["rmse"], 0.0)


def test_paper_metric_backend():
    generator = np.random.default_rng(3)
    reference = generator.random((8, 16, 16), dtype=np.float32)
    estimate = np.clip(reference * 0.9 + generator.random(reference.shape) * 0.02, 0, 1)
    metrics = fusion_metrics(reference, estimate, ratio=4, backend="paper")
    assert set(metrics) == {"psnr", "ssim", "sam", "ergas", "rmse"}
    assert metrics["rmse"] > 0


if __name__ == "__main__":
    test_patch_alignment_and_degradation()
    test_identical_metrics()
    test_paper_metric_backend()
    print("DATA_METRIC_TESTS_OK")
