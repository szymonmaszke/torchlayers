import torch

import pytest
import torchlayers


@pytest.fixture
def model():
    return torchlayers.Sequential(
        torchlayers.Conv(64),
        torchlayers.ReLU(),
        torchlayers.MaxPool(),
        torchlayers.Residual(
            torchlayers.Sequential(
                torchlayers.Conv(64, groups=16),
                torchlayers.ReLU(),
                torchlayers.BatchNorm(),
                torchlayers.Conv(64, groups=16),
                torchlayers.ChannelShuffle(groups=16),
                torchlayers.ReLU(),
            )
        ),
        torchlayers.SqueezeExcitation(),
        torchlayers.Sequential(
            torchlayers.Dropout(),
            torchlayers.Conv(128),
            torchlayers.ReLU(),
            torchlayers.InstanceNorm(),
        ),
        torchlayers.Poly(
            torchlayers.WayPoly(
                torchlayers.Fire(128),
                torchlayers.Fire(128),
                torchlayers.Fire(128),
                torchlayers.Fire(128),
            ),
            order=2,
        ),
        torchlayers.AvgPool(),
        torchlayers.StochasticDepth(torchlayers.Fire(128, hidden_channels=64)),
        torchlayers.ReLU(),
        torchlayers.GlobalAvgPool(),
        torchlayers.Linear(64),
    )


def test_functionality(model):
    model(torch.randn(16, 3, 28, 28))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(16):
        output = model(torch.randn(16, 3, 28, 28))
        loss = criterion(output, torch.randint(10, (16,)))
        loss.backward()

        optimizer.zero_grad()
