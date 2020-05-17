import typing

import torch

from . import module


class _GlobalPool(torch.nn.Module):
    def forward(self, inputs):
        return self._pooling(inputs).reshape(inputs.shape[0], -1)


class GlobalMaxPool1d(_GlobalPool):
    """Applies a 1D global max pooling over the last dimension.

    Usually used after last `Conv1d` layer to get maximum feature values
    for each timestep.

    Internally operates as `torch.nn.AdaptiveMaxPool1d` with redundant `1` dimensions
    flattened.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__()
        self._pooling = torch.nn.AdaptiveMaxPool1d(1)


class GlobalMaxPool2d(_GlobalPool):
    """Applies a 2D global max pooling over the last dimension(s).

    Usually used after last `Conv2d` layer to get maximum value feature values
    for each channel. Can be used on `3D` or `4D` input (though the latter is more common).

    Internally operates as `torch.nn.AdaptiveMaxPool2d` with redundant `1` dimensions
    flattened.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__()
        self._pooling = torch.nn.AdaptiveMaxPool2d(1)


class GlobalMaxPool3d(_GlobalPool):
    """Applies a 3D global max pooling over the last dimension(s).

    Usually used after last `Conv3d` layer to get maximum value feature values
    for each channel. Can be used on `4D` or `5D` input (though the latter is more common).

    Internally operates as `torch.nn.AdaptiveMaxPool3d` with redundant `1` dimensions
    flattened.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__()
        self._pooling = torch.nn.AdaptiveMaxPool3d(1)


class GlobalAvgPool1d(_GlobalPool):
    """Applies a 1D global average pooling over the last dimension.

    Usually used after last `Conv1d` layer to get mean of features values
    for each timestep.

    Internally operates as `torch.nn.AdaptiveAvgPool1d` with redundant `1` dimensions
    flattened.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__()
        self._pooling = torch.nn.AdaptiveAvgPool1d(1)


class GlobalAvgPool2d(_GlobalPool):
    """Applies a 2D global average pooling over the last dimension(s).

    Usually used after last `Conv2d` layer to get mean value of features values
    for each channel. Can be used on `3D` or `4D` input (though the latter is more common).

    Internally operates as `torch.nn.AdaptiveAvgPool3d` with redundant `1` dimensions
    flattened.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__()
        self._pooling = torch.nn.AdaptiveAvgPool2d(1)


class GlobalAvgPool3d(_GlobalPool):
    """Applies a 3D global average pooling over the last dimension(s).

    Usually used after last `Conv3d` layer to get mean value of features values
    for each channel. Can be used on `4D` or `5D` input (though the latter is more common).

    Internally operates as `torch.nn.AdaptiveAvgPool3d` with redundant `1` dimensions
    flattened.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__()
        self._pooling = torch.nn.AdaptiveAvgPool3d(1)


class GlobalMaxPool(module.InferDimension):
    """Perform `max` pooling operation leaving maximum values from channels.

    Usually used after last convolution layer (`torchlayers.Conv`)
    to get pixels of maximum value from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D` `GlobalMaxPool`
    will be used for `3D`, `4D` and `5D` shape respectively (batch included).

    Internally operates as `torchlayers.pooling.GlobalMaxPoolNd`.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__(
            dispatcher={5: GlobalMaxPool3d, 4: GlobalMaxPool2d, 3: GlobalMaxPool1d}
        )


class GlobalAvgPool(module.InferDimension):
    """Perform `mean` pooling operation leaving average values from channels.

    Usually used after last convolution layer (`torchlayers.Conv`) to get mean
    of pixels from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D`
    pooling will be used for `3D`, `4D` and `5D`
    shape respectively (batch included).

    Internally operates as `torchlayers.pooling.GlobalAvgPoolNd`.

    Returns
    -------
    `torch.Tensor`
        `2D` tensor `(batch, features)`

    """

    def __init__(self):
        super().__init__(
            dispatcher={5: GlobalAvgPool3d, 4: GlobalAvgPool2d, 3: GlobalAvgPool1d}
        )


class MaxPool(module.InferDimension):
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
            dispatcher={
                5: torch.nn.MaxPool3d,
                4: torch.nn.MaxPool2d,
                3: torch.nn.MaxPool1d,
            },
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )


class AvgPool(module.InferDimension):
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
            dispatcher={
                5: torch.nn.AvgPool3d,
                4: torch.nn.AvgPool2d,
                3: torch.nn.AvgPool1d,
            },
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
