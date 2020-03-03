import typing

import torch

from ._dev_utils.infer import create_forward, create_repr


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom in
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input.

    Parameters
    ----------
    function: Callable
        Any user specified function

    Returns
    -------
    Any
        Anything `function` returns

    """

    def __init__(self, function: typing.Callable):
        super().__init__()
        self.function: typing.Callable = function

    def forward(self, *args, **kwargs) -> typing.Any:
        return self.function(*args, **kwargs)


class Concatenate(torch.nn.Module):
    """Concatenate list of tensors.

    Mainly useful in `torch.nn.Sequential` when previous layer returns multiple
    tensors, e.g.:


        class Foo(torch.nn.Module):
            # Return same tensor three times
            # You could explicitly return a list or tuple as well
            def forward(tensor):
                return tensor, tensor, tensor


        model = torch.nn.Sequential(Foo(), torchlayers.Concatenate())
        model(torch.randn(10, 20))

    All tensors must have the same shape (except in the concatenating dimension).

    Parameters
    ----------
    dim: int
        Dimension along which tensors will be concatenated

    Returns
    -------
    torch.Tensor
        Concatenated tensor

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim

    def forward(self, tensor):
        return torch.cat(tensor, dim=self.dim)
