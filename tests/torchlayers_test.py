import pathlib
import tempfile

import torch

import pytest
import torchlayers as tl


@pytest.fixture
def model():
    return tl.Sequential(
        tl.Conv(64),
        tl.BatchNorm(),
        tl.ReLU(),
        tl.Conv(128),
        tl.BatchNorm(),
        tl.ReLU(),
        tl.Conv(256),
        tl.GlobalMaxPool(),
        tl.Linear(64),
        tl.BatchNorm(),
        tl.Linear(10),
    )


def test_functionality(model):
    # Initialize
    model = tl.build(model, torch.randn(16, 3, 28, 28))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(16):
        output = model(torch.randn(16, 3, 28, 28))
        loss = criterion(output, torch.randint(2, (16,)))
        loss.backward()

        optimizer.zero_grad()


def test_print_pre_init(model):
    target = r"""Sequential(
  (0): Conv(in_channels=?, out_channels=64, kernel_size=3, stride=1, padding=same, dilation=1, groups=1, bias=True, padding_mode=zeros)
  (1): BatchNorm(num_features=?, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Conv(in_channels=?, out_channels=128, kernel_size=3, stride=1, padding=same, dilation=1, groups=1, bias=True, padding_mode=zeros)
  (4): BatchNorm(num_features=?, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU()
  (6): Conv(in_channels=?, out_channels=256, kernel_size=3, stride=1, padding=same, dilation=1, groups=1, bias=True, padding_mode=zeros)
  (7): GlobalMaxPool()
  (8): Linear(in_features=?, out_features=64, bias=True)
  (9): BatchNorm(num_features=?, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (10): Linear(in_features=?, out_features=10, bias=True)
)"""

    assert target == str(model)


def test_attribute_access_existing():
    layer = tl.Conv(64)
    assert layer.kernel_size == 3
    assert layer.padding == "same"


def test_attribute_access_notinstantiated():
    layer = tl.Conv(64)
    with pytest.raises(AttributeError):
        non_instantiated_channels = layer.in_channels

    layer(torch.randn(1, 8, 28, 28))
    assert layer.in_channels == 8
