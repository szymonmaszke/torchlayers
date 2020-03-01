import itertools

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
        # torchlayers.InvertedResidualBottleneck(),
        torchlayers.AvgPool(),
        torchlayers.StochasticDepth(torchlayers.Fire(128, hidden_channels=64)),
        torchlayers.ReLU(),
        torchlayers.GlobalAvgPool(),
        torchlayers.Linear(64),
    )


def test_functionality(model):
    model = torchlayers.build(model, torch.randn(16, 3, 28, 28))
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(16):
        output = model(torch.randn(16, 3, 28, 28))
        loss = criterion(output, torch.randint(10, (16,)))
        loss.backward()

        optimizer.zero_grad()


# def test_same_padding_1d():
#     inputs = torch.randn(1, 5, 125)
#     for kernel_size, stride, dilation in itertools.product(
#         range(1, 20, 2), *[range(1, 20, 2) for _ in range(2)]
#     ):
#         model = torchlayers.build(
#             torchlayers.Conv(
#                 5,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 dilation=dilation,
#                 padding="same",
#             ),
#             inputs,
#         )
#         output = model(inputs)
#         assert output.shape == inputs.shape


# def test_same_padding_2d():
#     inputs = torch.randn(1, 5, 100, 100)
#     for kernel_size, stride, dilation in itertools.product(
#         range(1, 15, 2), *[range(1, 8, 2) for _ in range(2)]
#     ):
#         for kernel_size2, stride2, dilation2 in itertools.product(
#             range(1, 8, 2), *[range(1, 8, 2) for _ in range(2)]
#         ):
#             model = torchlayers.build(
#                 torchlayers.Conv(
#                     5,
#                     kernel_size=(kernel_size, kernel_size2),
#                     stride=(stride, stride2),
#                     dilation=(dilation, dilation2),
#                     padding="same",
#                 ),
#                 inputs,
#             )
#             output = model(inputs)
#             assert output.shape == inputs.shape
