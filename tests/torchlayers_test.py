import torch

import torchlayers


def test_conv_shape():
    layer = torchlayers.Conv(64, 3)
    assert layer(torch.randn(1, 3, 28, 28)).shape == (1, 64, 28, 28)


def test():
    model = torchlayers.Sequential(
        torchlayers.Conv(64),
        torchlayers.BatchNorm(),
        torchlayers.ReLU(),
        torchlayers.Conv(128),
        torchlayers.BatchNorm(),
        torchlayers.ReLU(),
        torchlayers.Conv(256),
        torchlayers.GlobalMaxPool(),
        torchlayers.Linear(64),
    )

    model(torch.randn(1, 3, 28, 28))

    print(model)

    assert 0 == 1
