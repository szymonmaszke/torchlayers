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
        return "{}()".format(type(self).__name__)

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
        `2D` tensor `(batch, features)`

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
        `2D` tensor `(batch, features)`

    """


class MaxPool(_dev_utils.modules.InferDimension):
    """Perform `max` operation across first `torch.Tensor` dimension.

    Depending on shape of passed `torch.Tensor` either `torch.nn.MaxPool1D`,
    `torch.nn.MaxPool2D` or `torch.nn.MaxPool3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Default value for `kernel_size` (`2`) was added.

    Parameters
    ----------
    kernel_size: int, optional
        The size of the window to take a max over. Default: `2`
    stride: int, optional
        The stride of the window. Default value is :attr:`kernel_size`
    padding: int, optional
        Implicit zero padding to be added on both sides. Default: `0`
    dilation: int
        Parameter controlling the stride of elements in the window. Default: `1`
    return_indices: bool, optional
        If ``True``, will return the max indices along with the outputs.
        Useful for :class:`torch.nn.MaxUnpool` later. Default: `False`
    ceil_mode: bool, optional
        When True, will use `ceil` instead of `floor` to compute the output shape.
        Default: `False`

    Returns
    -------
    `torch.Tensor`
        Same shape as `input` with values pooled.

    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
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
    """Perform `avg` operation across first `torch.Tensor` dimension.

    Depending on shape of passed `torch.Tensor` either `torch.nn.AvgPool1D`,
    `torch.nn.AvgPool2D` or `torch.nn.AvgPool3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Default value for `kernel_size` (`2`) was added.

    Parameters
    ----------
    kernel_size: int, optional
        The size of the window. Default: `2`
    stride: int, optional
        The stride of the window. Default value is :attr:`kernel_size`
    padding: int, oprtional
        Implicit zero padding to be added on both sides. Default: `0`
    ceil_mode: bool, opriontal
        When True, will use `ceil` instead of `floor` to compute the output shape.
        Default: `True`
    count_include_pad: bool, optional
        When True, will include the zero-padding in the averaging. Default: `True`

    Returns
    -------
    `torch.Tensor`
        Same shape as `input` with values pooled.

    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
