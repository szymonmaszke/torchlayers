![torchlayers Logo](https://github.com/szymonmaszke/torchlayers/blob/master/assets/banner.png)

--------------------------------------------------------------------------------

| Version | Docs | Tests | Coverage | Style | PyPI | Python | PyTorch |
|---------|------|-------|----------|-------|------|--------|---------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.1.0&color=377EF0&style=for-the-badge)](https://github.com/szymonmaszke/torchlayers/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://szymonmaszke.github.io/torchlayers/)  | ![Tests](https://github.com/szymonmaszke/torchlayers/workflows/test/badge.svg) | ![Coverage](https://img.shields.io/codecov/c/github/szymonmaszke/torchlayers?label=%20&logo=codecov&style=for-the-badge) | [![codebeat](https://img.shields.io/static/v1?label=&message=CB&color=27A8E0&style=for-the-badge)](https://codebeat.co/projects/github-com-szymonmaszke-torchlayers-master) | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchlayers/) | [![Python](https://img.shields.io/static/v1?label=&message=>3.5&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=>=1.3.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Docker](https://img.shields.io/static/v1?label=&message=docker&color=309cef&style=for-the-badge)](https://cloud.docker.com/u/szymonmaszke/repository/docker/szymonmaszke/torchlayers) |

[__torchlayers__](https://szymonmaszke.github.io/torchlayers/) is a library based on [__PyTorch__](https://pytorch.org/)
providing __automatic shape and dimensionality inference of `torch.nn` layers__ + additional
building blocks featured in current SOTA architectures (e.g. [Efficient-Net](https://arxiv.org/abs/1905.11946)).

Above requires no user intervention (except single call to `torchlayers.build`)
similarly to the one seen in [__Keras__](https://www.tensorflow.org/guide/keras).

### Main functionalities:

* __Shape inference__ for most of `torch.nn` module (__convolutional, recurrent, transformer, attention and linear layers__)
* __Shape inference of custom modules__ (see examples section)
* __Dimensionality inference__ (e.g. `torchlayers.Conv` working as `torch.nn.Conv1d/2d/3d` based on input shape)
* __Additional [Keras-like](https://www.tensorflow.org/guide/keras) layers__ (e.g. `torchlayers.Reshape` or `torchlayers.StandardNormalNoise`)
* __Additional SOTA layers__ mostly from ImageNet competitions
(e.g. [PolyNet](https://arxiv.org/abs/1608.06993),
[Squeeze-And-Excitation](https://arxiv.org/abs/1709.01507),
[StochasticDepth](www.arxiv.org/abs/1512.03385>))
* __Zero overhead and [torchscript](https://pytorch.org/docs/stable/jit.html) support__

# Examples

For full functionality please check [__torchlayers documentation__](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge).
Below examples should introduce all necessary concepts you should know.

## Simple convolutional image/text classifier

* Yes, one model can be instantiated to do both easily.
First define model using `torch.nn` and `torchlayers`:

```python
import torch
import torchlayers

# torch.nn and torchlayers can be mixed easily
model = torch.nn.Sequential(
    torchlayers.Conv(64),  # specify ONLY out_channels
    torch.nn.ReLU(),  # use torch.nn wherever you wish
    torchlayers.BatchNorm(),  # BatchNormNd inferred from input
    torchlayers.Conv(128),  # Default kernel_size equal to 3
    torchlayers.ReLU(),
    torchlayers.Conv(256, kernel_size=11),  # "same" padding as default
    torchlayers.GlobalMaxPool(),  # Known from Keras
    torchlayers.Linear(10),  # Output for 10 classes
)
```

Above would give you model's summary like this:

```python
# Insert output here
```

* Now you can __build__/instantiate your model with example input (in this case MNIST-like input):

```python
mnist_model = torchlayers.build(model, torch.randn(1, 3, 28, 28))
```

Or if it's textual data you are after, same model can be built with different
input shape (e.g. for text classification using `300` dimensional pretrained embedding):

```python
# [batch, embedding, timesteps], first and last dimensions can be any
text_model = torchlayers.build(model, torch.randn(1, 300, 1))
```

Finally, you can `print` both models after instantiation, here provided side-by-side for comparison

```python
# Insert output
```

## Custom modules with shape inference capabilities

User can define any module and make it shape inferable, provided it's first
argument is `int` and can be inferred from first `tensor` element

## Autoencoder with inverted residual bottleneck and pixel shuffle

We will strive for example model for `ImageNet` reconstruction.
Think of it like a demonstration of capabilities for different layers
and architectures used for vision tasks.

Please check comments and documentation to follow this example easier:

```python
# Input - 3 x 256 x 256 for ImageNet reconstruction
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        self.encoder = torchlayers.Sequential(
            torchlayers.Conv(64, kernel_size=7),
            torchlayers.activations.Swish(),  # Direct access to module .activations
            torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
            torchlayers.AvgPool(),  # shape 64 x 128 x 128, kernel_size=2 by default
            torchlayers.HardSwish(),  # Access simply through torchlayers
            torchlayers.SeparableConv(128),  # Up number of channels to 128
            torchlayers.InvertedResidualBottleneck(),  # Default with squeeze excitation
            torch.nn.ReLU(),
            torchlayers.AvgPool(),  # shape 128 x 64 x 64, kernel_size=2 by default
            torchlayers.DepthwiseConv(256),  # Depthwise convolution even easier to use
            # Pass input thrice through the same weights like in PolyNet
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.ReLU(),  # all torch.nn can be accessed via torchlayers
            torchlayers.MaxPool(),  # shape 256 x 32 x 32
            torchlayers.Fire(out_channels=512),  # shape 512 x 32 x 32
            torchlayers.SqueezeExcitation(hidden=64),
            torchlayers.InvertedResidualBottleneck(),
            torchlayers.MaxPool(),  # shape 512 x 16 x 16
            torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
            # Randomly Switch the last two layers with 0.5 probability
            torchlayers.StochasticDepth(
                torch.nn.Sequential(
                    torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
                    torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
                ),
                p=0.5,
            ),
            torchlayers.AvgPool(),  # shape 512 x 8 x 8
        )

        self.decoder = torchlayers.Sequential(
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=2),
            # Has ICNR initialization by default as well
            torchlayers.Conv2dPixelShuffle(out_channels=512, upscale_factor=2),
            # Shape 512 x 16 x 16 after PixelShuffle
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.Conv2dPixelShuffle(out_channels=256, upscale_factor=2),
            # Shape 256 x 32 x 32
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.Conv2dPixelShuffle(out_channels=128, upscale_factor=2),
            # Shape 128 x 64 x 64
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=4),
            torchlayers.Conv2dPixelShuffle(out_channels=64, upscale_factor=2),
            # Shape 64 x 128 x 128
            torchlayers.InvertedResidualBottleneck(),
            torchlayers.Conv(256),
            torchlayers.Swish(),
            torchlayers.BatchNorm(),
            torchlayers.Conv2dPixelShuffle(out_channels=32, upscale_factor=2),
            # Shape 32 x 256 x 256
            torchlayers.Conv(16),
            torchlayers.Swish(),
            torchlayers.Conv(3),
            # Shape 3 x 256 x 256
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))

```

# Installation

## [pip](<https://pypi.org/project/torchlayers/>)

### Latest release:

```shell
pip install --user torchlayers
```

### Nightly:

```shell
pip install --user torchlayers-nightly
```

## [Docker](https://cloud.docker.com/repository/docker/szymonmaszke/torchlayers)

__CPU standalone__ and various versions of __GPU enabled__ images are available
at [dockerhub](https://cloud.docker.com/repository/docker/szymonmaszke/torchlayers).

For CPU quickstart, issue:

```shell
docker pull szymonmaszke/torchlayers:18.04
```

Nightly builds are also available, just prefix tag with `nightly_`. If you are going for `GPU` image make sure you have
[nvidia/docker](https://github.com/NVIDIA/nvidia-docker) installed and it's runtime set.
