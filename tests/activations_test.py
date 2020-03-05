import torch

import torchlayers


def test_hardsigmoid():
    torchlayers.HardSigmoid()(torch.randn(4, 5, 6))
