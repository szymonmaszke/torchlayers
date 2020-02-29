import typing

import torch

from . import _dev_utils


class _GlobalPool(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._operation = self._maximum if "Max" in type(self).__name__ else self._mean

    def _mean(self, tensor):
        return torch.mean(tensor, axis=-1)

    def _maximum(self, tensor):
        values, _ = torch.max(tensor, axis=-1)
        return values

    def __repr__(self):
        return f"{type(self).__name__}()"

    def forward(self, inputs):
        while len(inputs.shape) > 2:
            inputs = self._operation(inputs)
        return inputs


class GlobalMaxPool(_GlobalPool):
    """Perform `max` operation across first `torch.Tensor` dimension.

    Usually used after last convolution layer to get pixels of maximum value
    from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Returns
    -------
    `torch.Tensor`
            `2D` tensor `(batch, channels)`

    """


class GlobalAvgPool(_GlobalPool):
    """Perform `mean` operation across first `torch.Tensor` dimension.

    Usually used after last convolution layer to get mean of pixels
    from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Returns
    -------
    `torch.Tensor`
            `2D` tensor `(batch, channels)`

    """


class MaxPool(_dev_utils.modules.InferDimension):
    """Perform `max` operation across first `torch.Tensor` dimension.

    Usually used after last convolution layer to get pixels of maximum value
    from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Returns
    -------
    `torch.Tensor`
            Same shape as `input`. Acts just like `GlobalMaxPool` but does not remove
            superficial `1` dimensions. Works exactly like `AdaptiveMaxPool` in PyTorch
            except for dimension inferring.

    """

    def __init__(
        self,
        kernel_size=2,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )


class AvgPool(_dev_utils.modules.InferDimension):
    """Perform `max` operation across first `torch.Tensor` dimension.

    Usually used after last convolution layer to get pixels of maximum value
    from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Returns
    -------
    `torch.Tensor`
            Same shape as `input`. Acts just like `GlobalMaxPool` but does not remove
            superficial `1` dimensions. Works exactly like `AdaptiveMaxPool` in PyTorch
            except for dimension inferring.

    """

    def __init__(
        self,
        kernel_size=2,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
