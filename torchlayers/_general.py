import typing

import torch

from ._dev_utils.infer import create_forward, create_repr


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom functions within
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input.

    Parameters
    ----------
    function: typing.Callable
            Any argument user specified function

    Returns
    -------
    typing.Any
            Anything `function` returns

    """

    def __init__(self, function: typing.Callable):
        super().__init__()
        self.function: typing.Callable = function

    def forward(self, *args, **kwargs) -> typing.Any:
        return self.function(*args, **kwargs)


class Flatten(torch.nn.Module):
    """Flatten input (except batch dimension).

    Returns
    -------
    torch.Tensor
            Flattened input `torch.Tensor`.

    """

    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)
