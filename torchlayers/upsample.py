import typing

import torch

from . import convolution


class ConvPixelShuffle(torch.nn.Module):
    """Two dimensional convolution with ICNR initialization followed by PixelShuffle.

    Increases `height` and `width` of `input` tensor by scale, acts like
    learnable upsampling. Due to `ICNR weight initialization <https://arxiv.org/abs/1707.02937>`__
    of `convolution` it has similar starting point to nearest neighbour upsampling.

    `kernel_size` got a default value of `3`, `upscale_factor` got a default
    value of `2`.

    Example::

        import torchlayers as tl


        class MiniAutoEncoder(tl.Module):
            def __init__(self, out_channels):
                super().__init__()
                self.conv1 = tl.Conv(64)
                # Twice smaller image by default
                self.pooling = tl.MaxPool()
                # Twice larger (upscale_factor=2) by default
                self.upsample = tl.ConvPixelShuffle(out_channels)

            def forward(self, x):
                x = self.conv1(x)
                pooled = self.pooling(x)
                return self.upsample(pooled)


        out_channels = 3
        network = MiniAutoEncoder(out_channels)
        tl.build(network, torch.randn(1, out_channels, 64, 64))
        assert network(torch.randn(5, out_channels, 64, 64)).shape == [5, out_channels, 64, 64]

    .. note::

        Currently only `4D` input is allowed (`[batch, channels, height, width]`),
        due to `torch.nn.PixelShuffle` not supporting `3D` or `5D` inputs.
        See [this PyTorch PR](https://github.com/pytorch/pytorch/pull/6340/files)
        for example of dimension-agnostic implementation.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced after PixelShuffle
    upscale_factor : int, optional
        Factor to increase spatial resolution by. Default: `2`
    kernel_size : int or tuple, optional
        Size of the convolving kernel. Default: `3`
    stride : int or tuple, optional
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
    initializer: typing.Callable[[torch.Tensor,], torch.Tensor], optional
        Initializer for ICNR initialization, can be a function from `torch.nn.init`.
        Receive tensor as argument and returns tensor after initialization.
        Default: `torch.nn.init.kaiming_normal_`

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_factor: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding="same",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        initializer=None,
    ):
        super().__init__()
        self.convolution = convolution.Conv(
            in_channels,
            out_channels * upscale_factor * upscale_factor,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        self.upsample = torch.nn.PixelShuffle(upscale_factor)
        if initializer is None:
            self.initializer = torch.nn.init.kaiming_normal_
        else:
            self.initializer = initializer

    def post_build(self):
        """Initialize weights after layer was built."""
        self.icnr_initialization(self.convolution.weight.data)

    def icnr_initialization(self, tensor):
        """ICNR initializer for checkerboard artifact free sub pixel convolution.

        Originally presented in
        `Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize <https://arxiv.org/abs/1707.02937>`__
        Initializes convolutional layer prior to `torch.nn.PixelShuffle`.
        Weights are initialized according to `initializer` passed to to `__init__`.

        Parameters
        ----------
        tensor: torch.Tensor
                Tensor to be initialized using ICNR init.

        Returns
        -------
        torch.Tensor
                Tensor initialized using ICNR.

        """

        if self.upsample.upscale_factor == 1:
            return self.initializer(tensor)

        new_shape = [int(tensor.shape[0] / (self.upsample.upscale_factor ** 2))] + list(
            tensor.shape[1:]
        )

        subkernel = self.initializer(torch.zeros(new_shape)).transpose(0, 1)

        kernel = subkernel.reshape(subkernel.shape[0], subkernel.shape[1], -1).repeat(
            1, 1, self.upsample.upscale_factor ** 2
        )

        return kernel.reshape([-1, tensor.shape[0]] + list(tensor.shape[2:])).transpose(
            0, 1
        )

    def forward(self, inputs):
        return self.upsample(self.convolution(inputs))
