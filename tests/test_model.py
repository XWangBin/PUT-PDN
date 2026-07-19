import torch

from put_pdn.inference import predict
from put_pdn.models import PUT


def test_put_forward_shapes():
    model = PUT(ratio=4, out_channels=8, channels=8, stage=1)
    lr_hsi = torch.rand(1, 8, 4, 4)
    pan = torch.rand(1, 1, 16, 16)
    output, reconstructed_pan, reconstructed_lr_hsi = model(lr_hsi, pan)
    assert output.shape == (1, 8, 16, 16)
    assert reconstructed_pan.shape == pan.shape
    assert reconstructed_lr_hsi.shape == lr_hsi.shape


def test_tiled_prediction_shape():
    model = PUT(ratio=4, out_channels=8, channels=8, stage=1)
    lr_hsi = torch.rand(1, 8, 8, 8)
    pan = torch.rand(1, 1, 32, 32)
    output = predict(model, lr_hsi, pan, ratio=4, tile_size=16, overlap=8)
    assert output.shape == (1, 8, 32, 32)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_put_forward_shapes()
    test_tiled_prediction_shape()
    print("MODEL_TESTS_OK")
