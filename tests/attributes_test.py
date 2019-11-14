import torch

import torchlayers


# User __dir__ for module iteration
def test_module():
    layer = torchlayers.Conv2d(64, kernel_size=3)
    assert layer.__module__ == "torchlayers"
