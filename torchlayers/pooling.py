import torch

from ._dev_utils import modules


class _GlobalPool(modules.InferDimension):
    @classmethod
    def _squeeze(cls, inputs):
        squeezed = inputs.squeeze()
        # Batch dimension could be squeezed as well
        if len(squeezed.shape) == 1:
            return squeezed.unsqueeze(0)
        return squeezed

    def __init__(self):
        _operation: str = "Max" if "Max" in type(self).__name__ else "Avg"
        super().__init__(output_size=1)
        self._module_name = "Adaptive" + _operation + "Pool"

    def __repr__(self):
        return f"{type(self).__name__}()"

    def forward(self, inputs):
        return _GlobalPool._squeeze(super().forward(inputs))


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


class MaxPool(modules.InferDimension):
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


class AvgPool(modules.InferDimension):
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
