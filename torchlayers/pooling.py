import torch

__all__ = ["GlobalMaxPool", "GlobalAvgPool", "AdaptiveMaxPool", "AdaptiveAvgPool"]


class _GlobalPool(torch.nn.Module):
    def __init__(self, is_max: bool, squeeze: bool):
        super().__init__()
        if is_max:
            self._pool_operation_name = "max"
        else:
            self._pool_operation_name = "avg"
        self._squeeze = squeeze

    def forward(self, inputs):
        if not hasattr(self, "_pool_operation"):
            module = getattr(
                torch.nn.functional,
                f"adaptive_{self._pool_operation_name}_pool{len(inputs.shape) - 2}d",
                None,
            )
            if module is None:
                raise ValueError(
                    f"{type(self).__name__} could not be inferred from shape. "
                    + f"Only 5, 4 and 3 dimensional input, got {len(inputs.shape)}"
                )
            self._pool_operation = module

        output = self._pool_operation(inputs, 1)
        if self._squeeze:
            squeezed = output.squeeze()
            if len(squeezed.shape) == 1:
                return squeezed.unsqueeze(0)
        return output


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

    def __init__(self):
        super().__init__(True, squeeze=True)


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

    def __init__(self):
        super().__init__(False, squeeze=True)


class AdaptiveMaxPool(_GlobalPool):
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

    def __init__(self):
        super().__init__(True, squeeze=False)


class AdaptiveAvgPool(_GlobalPool):
    """Perform `mean` operation across first `torch.Tensor` dimension.

    Usually used after last convolution layer to get pixels of maximum value
    from each channel.

    Depending on shape of passed `torch.Tensor` either `1D`, `2D` or `3D` pooling will be used
    for `3D`, `4D` and `5D` shape respectively (batch included).

    Acts just like `GlobalMaxPool` but does not remove
    superficial `1` dimensions. Works exactly like `AdaptiveAvgPool` in PyTorch
    except for dimension inferring.

    Returns
    -------
    `torch.Tensor`
            Same shape as `input`.

    """

    def __init__(self):
        super().__init__(False, squeeze=False)
