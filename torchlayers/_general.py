import typing

import torch

from ._dev_utils.infer import create_forward, create_repr


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom in
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input.

    Parameters
    ----------
    function : Callable
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
        model(torch.randn(64 , 20))

    All tensors must have the same shape (except in the concatenating dimension).

    Parameters
    ----------
    dim : int
        Dimension along which tensors will be concatenated

    Returns
    -------
    torch.Tensor
        Concatenated tensor along specified `dim`.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Reshape(torch.nn.Module):
    """Reshape tensor excluding `batch` dimension

    Reshapes input `torch.Tensor` features while preserving batch dimension.
    Standard `torch.reshape` values (e.g. `-1`) are supported, e.g.:


        layer = torchlayers.Reshape(20, -1)
        layer(torch.randn(64, 80)) # shape (64, 20, 4)

    All tensors must have the same shape (except in the concatenating dimension).
    If possible, no copy of `tensor` will be performed.

    Parameters
    ----------
    shapes: *int
        Variable length list of shapes used in view function

    Returns
    -------
    torch.Tensor
        Concatenated tensor

    """

    def __init__(self, *shapes: int):
        super().__init__()
        self.shapes: typing.Tuple[int] = shapes

    def forward(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0], *self.shapes))
