import itertools

import torch

import pytest
import torchlayers as tl


def single_module(regularization: str, name: str):
    return getattr(tl, regularization)(tl.Linear(10), weight_decay=1000000, name=name)


def multiple_modules(regularization: str, name: str):
    return getattr(tl, regularization)(
        torch.nn.Sequential(
            tl.Linear(40), tl.ReLU(), tl.Linear(20), tl.ReLU(), tl.Linear(10)
        ),
        weight_decay=1000000,
        name=name,
    )


def generate_inputs():
    modules = (single_module, multiple_modules)
    names = ("bias", "weight", None)
    regularizations = ("L1", "L2")
    steps = (1, 3)
    for module, regularization, name, steps in itertools.product(
        modules, regularizations, names, steps
    ):
        yield module(regularization, name), name, steps


def check_gradient_correctness(module, name):
    for param_name, param in module.named_parameters():
        if name is not None:
            if name in param_name:
                assert torch.abs(param.grad).sum() > 10000.0
            else:
                assert torch.abs(param.grad).sum() < 10000.0
        else:
            assert torch.abs(param.grad).sum() > 10000.0


@pytest.mark.parametrize("module,name,steps", list(generate_inputs()))
def test_weight_decay(module, name, steps: bool):
    tl.build(module, torch.randn(1, 20))
    for _ in range(steps):
        output = module(torch.randn(5, 20))
        output.sum().backward()
    check_gradient_correctness(module, name)
    module.zero_grad()
    output = module(torch.rand(5, 20))
    output.sum().backward()
    check_gradient_correctness(module, name)
