import collections
import itertools
import math
import typing

import torch

from ._dev_utils import modules
from .pooling import GlobalAvgPool


class Conv(modules.InferDimension):
    """Standard convolution layer.

    Based on input shape it either creates 1D, 2D or 3D convolution for inputs of shape
    3D, 4D, 5D respectively (including batch as first dimension).

    Additional `same` `padding` mode was added and set as default. Using it input dimensions
    (except for channels) like height and width will be preserved (for odd kernel sizes).

    `kernel_size` got a default value of `3`.

    Otherwise acts exactly like PyTorch's Convolution, see
    `documentation <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__.

    Parameters
    ----------
    in_channels: int
        Number of channels in the input image
    out_channels: int
        Number of channels produced by the convolution
    kernel_size: int or tuple, optional
        Size of the convolving kernel. Default: 3
    stride: int or tuple, optional
        Stride of the convolution. Default: 1
    padding: int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0
    padding_mode: string, optional
        Accepted values `zeros` and `circular` Default: `zeros`
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups: int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(
            instance_creator=Conv._pad,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    @classmethod
    def _dimension_pad(cls, dimension, dilation, kernel_size, stride):
        if kernel_size % 2 == 0:
            raise ValueError(
                'Only odd kernel size for padding "same" is currently supported.'
            )

        return max(
            math.ceil(
                (dimension * stride - dimension + dilation * (kernel_size - 1)) // 2
            ),
            0,
        )

    @classmethod
    def _expand_if_needed(cls, dimensions, argument):
        if isinstance(argument, collections.abc.Iterable):
            return argument
        return tuple(itertools.repeat(argument, len(dimensions)))

    @classmethod
    def _pad(cls, inputs, inner_class, **kwargs):
        if isinstance(kwargs["padding"], str) and kwargs["padding"].lower() == "same":
            dimensions = inputs.shape[2:]
            paddings = tuple(
                cls._dimension_pad(dimension, dilation, kernel_size, stride)
                for dimension, dilation, kernel_size, stride in zip(
                    dimensions,
                    *[
                        cls._expand_if_needed(dimensions, kwargs[name])
                        for name in ("dilation", "kernel_size", "stride")
                    ],
                )
            )
            kwargs["padding"] = paddings

        return inner_class(**kwargs)


class ConvTranspose(modules.InferDimension):
    """Standard transposed convolution layer.

    Based on input shape it either creates 1D, 2D or 3D convolution (for inputs of shape
    3D, 4D, 5D including batch as first dimension).

    Otherwise acts exactly like PyTorch's Convolution, see
    `documentation <https://pytorch.org/docs/stable/nn.html#convolution-layers>`__.

    Parameters
    ----------
    in_channels: int
        Number of channels in the input image
    out_channels: int
        Number of channels produced by the convolution
    kernel_size: int or tuple
        Size of the convolving kernel
    stride: int or tuple, optional
        Stride of the convolution. Default: 1
    padding: int or tuple, optional
        ``dilation * (kernel_size - 1) - padding`` zero-padding
        will be added to both sides of the input. Default: 0
    output_padding: int or tuple, optional
        Additional size added to one side of the output shape. Default: 0
    groups: int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias: bool, optional
        If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    dilation: int or tuple, optional
        Spacing between kernel elements. Default: 1
    padding_mode: string, optional
        Accepted values `zeros` and `circular` Default: `zeros`

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )


class ChannelShuffle(modules.Representation):
    """Shuffle output channels from modules.

    When using group convolution knowledge transfer between next layers is reduced
    (as the same input channels are convolved with the same output channels).

    This layer reshuffles output channels via simple `reshape` in order to mix the representation
    from separate groups and improve knowledge transfer.

    Originally proposed by Xiangyu Zhang et. al in:
    `ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices <https://arxiv.org/abs/1707.01083>`__

    Parameters
    ----------
    groups: int
            Count of groups used in the previous convolutional layer.

    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups: int = groups

    def forward(self, inputs):
        return (
            inputs.reshape(inputs.shape[0], self.groups, -1, *inputs.shape[2:])
            .transpose(1, 2)
            .reshape(*inputs.shape)
        )


class ChannelSplit(modules.Representation):
    """Convenience layer splitting tensor using ratio.

    Returns two outputs, splitted accordingly to parameters.

    Parameters
    ----------
    ratio: float
            Percentage of channels to be split
    dim: int
            Dimension along which input will be splitted. Default: `1` (channel dimension)

    """

    def __init__(self, ratio: float, dim: int = 1):
        super().__init__()
        if not 0.0 < ratio < 1.0:
            raise ValueError(
                "Ratio of small expand fire module has to be between 0 and 1."
            )

        self.ratio: float = ratio
        self.dim: int = dim

    def forward(self, inputs):
        return torch.split(inputs, int(inputs.shape[1] * self.ratio), dim=self.dim)


class Residual(torch.nn.Module):
    """Residual connection adding input to output of provided module.

    Originally proposed by He et. al in `ResNet <www.arxiv.org/abs/1512.03385>`__

    For correct usage it is advised to keep input line (skip connection) without
    any layer or activation and implement transformations only in module arguments
    (as per https://arxiv.org/pdf/1603.05027.pdf).

    Above can be easily achieved by using one of BatchNormConv competitorch modules.

    Parameters
    ----------
    module: torch.nn.Module
            Convolutional PyTorch module (or other compatible module).
            Shape of module's `inputs` has to be equal to it's `outputs`, both
            should be addable `torch.Tensor` instances.

    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class Dense(torch.nn.Module):
    """Dense residual connection concatenating input channels and output channels of provided module.

    Originally proposed by Gao Huang et. al in `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`__

    Parameters
    ----------
    module: torch.nn.Module
            Convolutional PyTorch module (or other compatible module).
            Shape of module's `inputs` has to be equal to it's `outputs`, both
            should be addable `torch.Tensor` instances.
    dim: int, optional
            Dimension along which `input` and module's `output` will be concatenated.
            Default: `1` (channel-wise)

    """

    def __init__(self, module: torch.nn.Module, dim: int = 1):
        super().__init__()
        self.module: torch.nn.Module = module
        self.dim: int = dim

    def forward(self, inputs):
        return torch.cat(self.module(inputs), inputs, dim=self.dim)


class Poly(torch.nn.Module):
    """Apply one module to input multiple times and sum.

    It's equation for `order` equal to :math:`N` can be written as:

    .. math::
        1 + F + F^2 + ... + F^N

    where :math:`1` is identity and :math:`F` is mapping specified by `module`.

    Originally proposed by Xingcheng Zhang et. al in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks <https://arxiv.org/abs/1608.06993>`__

    Parameters
    ----------
    module: torch.nn.Module
            Convolutional PyTorch module (or other compatible module).
            `inputs` shape has to be equal to it's `output` shape
            (for 2D convolution it would be :math:`(C, H, W)` (channels, height, width respectively)).
    order: int, optional
            Order of PolyInception module. For order equal to `1` acts just like
            ResNet, order of `2` was used in original paper. Default: `2`

    """

    def __init__(self, module: torch.nn.Module, order: int = 2):
        super().__init__()
        if order < 1:
            raise ValueError("Order of Poly cannot be less than 1.")

        self.module: torch.nn.Module = module
        self.order: int = order

    def extra_repr(self):
        return f"order={self.order},"

    def forward(self, inputs):
        outputs = [self.module(inputs)]
        for _ in range(1, self.order):
            outputs.append(self.module(outputs[-1]))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


class MPoly(torch.nn.Module):
    """Apply multiple (m) modules to input multiple times and sum.

    It's equation for `modules` length equal to :math:`N` would be:

    .. math::
        1 + F_0 + F_0 * F_1 + ... + F_0 * F_1 * ... * F_N

    where :math:`1` is identity and consecutive :math:`F_N` are consecutive models
    specified by user.

    Originally proposed by Xingcheng Zhang et. al in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks <https://arxiv.org/abs/1608.06993>`__

    Parameters
    ----------
    modules: *torch.nn.Module
            Var arg with modules to use with WayPoly. If empty, acts as an identity.
            for one module, acts like `ResNet`. `2` were used in original paper.
            All modules need `inputs` and `outputs` shape equal and equal between themselves.

    """

    def __init__(self, *modules: torch.nn.Module):
        super().__init__()
        self.modules_: torch.nn.Module = torch.nn.ModuleList(modules)

    def forward(self, inputs):
        outputs = [self.modules_[0](inputs)]
        for module in self.modules_[1:]:
            outputs.append(self.module(outputs[-1]))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


class WayPoly(torch.nn.Module):
    """Apply multiple modules to input and sum.

    It's equation for `modules` length equal to :math:`N` would be:

    .. math::
        1 + F_1 + F_2 + ... + F_N

    where :math:`1` is identity and consecutive :math:`F_N` are consecutive models
    specified by user.

    Can be considered as an extension of standard `ResNet` to many modules.

    Originally proposed by Xingcheng Zhang et. al in
    `PolyNet: A Pursuit of Structural Diversity in Very Deep Networks <https://arxiv.org/abs/1608.06993>`__

    Parameters
    ----------
    modules: *torch.nn.Module
            Var arg with modules to use with WayPoly. If empty, acts as an identity.
            for one module, acts like `ResNet`. `2` was used in original paper.
            All modules need `inputs` and `outputs` shape equal and equal between themselves.
    """

    def __init__(self, *modules: torch.nn.Module):
        super().__init__()
        self.modules_: torch.nn.Module = torch.nn.ModuleList(modules)

    def forward(self, inputs):
        outputs = []
        for module in self.modules_:
            outputs.append(module(inputs))
        return torch.stack([inputs] + outputs, dim=0).sum(dim=0)


class SqueezeExcitation(modules.Representation):
    """Learn channel-wise excitation maps for `inputs`.

    Provided `inputs` will be squeezed into `in_channels`, passed through two
    non-linear layers, rescaled to :math:`[0, 1]` and multiplied with original input.

    Originally proposed by Xingcheng Zhang et. al in
    `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`__

    Parameters
    ----------
    in_channels: int
        Number of channels in the input
    hidden: int, optional
        Size of the hidden `torch.nn.Linear` layer. Usually smaller than `in_channels`
        (at least in original research paper). Default: Half of `in_channels`
    activation: typing.Callable, optional
        One argument callable performing activation after hidden layer.
        Default: `torch.nn.ReLU()`

    """

    def __init__(self, in_channels: int, hidden: int = None, activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden: int = hidden if hidden is not None else in_channels // 2

        self._pooling = GlobalAvgPool()
        self._first = torch.nn.Linear(in_channels, self.hidden)
        self.activation: typing.Callable = activation if activation is not None else torch.nn.ReLU()
        self._second = torch.nn.Linear(self.hidden, in_channels)

    def forward(self, inputs):
        excitation = torch.sigmoid(
            self._second(self.activation(self._first(self._pooling(inputs))))
        )
        for _ in range(len(inputs.shape) - 2):
            excitation = excitation.unsqueeze(-1)

        return inputs * excitation


class Fire(modules.Representation):
    """Squeeze and Expand number of channels efficiently operation-wise.

    First input channels will be squeezed to `hidden` channels and :math:`1 x 1` convolution.
    After that those will be expanded to `out_channels` partially done by :math:`3 x 3` convolution
    and partially by :math:`1 x 1` convolution (as specified by `ratio` parameter).

    Originally proposed by Forrest N. Iandola et. al in
    `SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size <https://arxiv.org/abs/1602.07360>`__

    Parameters
    ----------
    in_channels: int
        Number of channels in the input
    out_channels: int
        Number of channels produced by Fire module
    hidden: int, optional
        Number of hidden channels (squeeze convolution layer).
        Default: Half of `in_channels`
    ratio: float, optional
        Ratio of :math:`1 x 1` convolution taken from total `out_channels`.
        The more, the more :math:`1 x 1` convolution will be used during expanding.
        Default: `0.5` (half of `out_channels`)

    """

    def __init__(
        self, in_channels, out_channels, hidden_channels=None, ratio: float = 0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (
            hidden_channels if hidden_channels is not None else in_channels // 2
        )
        self.ratio: float = 0.5

        self._squeeze = torch.nn.Conv2d(
            in_channels, self.hidden_channels, kernel_size=1
        )

        small_out_channels = int(out_channels * self.ratio)
        self._expand_small = torch.nn.Conv2d(
            self.hidden_channels, small_out_channels, kernel_size=1
        )
        self._expand_large = torch.nn.Conv2d(
            self.hidden_channels,
            out_channels - small_out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, inputs):
        squeeze = self._squeeze(inputs)
        return torch.cat(
            (self._expand_small(squeeze), self._expand_large(squeeze)), dim=1
        )
