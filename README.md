![torchlayers Logo](https://github.com/szymonmaszke/torchlayers/blob/master/assets/banner.png)

--------------------------------------------------------------------------------


| Version | Docs | Tests | Coverage | Style | PyPI | Python | PyTorch | Docker |
|---------|------|-------|----------|-------|------|--------|---------|--------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.1.0&color=377EF0&style=for-the-badge)](https://github.com/szymonmaszke/torchlayers/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://szymonmaszke.github.io/torchlayers/)  | ![Tests](https://github.com/szymonmaszke/torchlayers/workflows/test/badge.svg) | [![codecov](https://codecov.io/gh/szymonmaszke/torchlayers/branch/master/graph/badge.svg?token=GbZmdqbTWM)](https://codecov.io/gh/szymonmaszke/torchlayers) | [![codebeat](https://img.shields.io/static/v1?label=&message=CB&color=27A8E0&style=for-the-badge)](https://codebeat.co/projects/github-com-szymonmaszke-torchlayers-master) | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchlayers/) | [![Python](https://img.shields.io/static/v1?label=&message=>=3.7&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=>=1.3.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Docker](https://img.shields.io/static/v1?label=&message=docker&color=309cef&style=for-the-badge)](https://cloud.docker.com/u/szymonmaszke/repository/docker/szymonmaszke/torchlayers) |

[__torchlayers__](https://szymonmaszke.github.io/torchlayers/) is a library based on [__PyTorch__](https://pytorch.org/)
providing __automatic shape and dimensionality inference of `torch.nn` layers__ + additional
building blocks featured in current SOTA architectures (e.g. [Efficient-Net](https://arxiv.org/abs/1905.11946)).

Above requires no user intervention (except single call to `torchlayers.build`)
similarly to the one seen in [__Keras__](https://www.tensorflow.org/guide/keras).

### Main functionalities:

* __Shape inference__ for most of `torch.nn` module (__convolutional, recurrent, transformer, attention and linear layers__)
* __Dimensionality inference__ (e.g. `torchlayers.Conv` working as `torch.nn.Conv1d/2d/3d` based on `input shape`)
* __Shape inference of custom modules__ (see examples section)
* __Additional [Keras-like](https://www.tensorflow.org/guide/keras) layers__ (e.g. `torchlayers.Reshape` or `torchlayers.StandardNormalNoise`)
* __Additional SOTA layers__ mostly from ImageNet competitions
(e.g. [PolyNet](https://arxiv.org/abs/1608.06993),
[Squeeze-And-Excitation](https://arxiv.org/abs/1709.01507),
[StochasticDepth](www.arxiv.org/abs/1512.03385>))
* __Useful defaults__ (`"same"` padding and default `kernel_size=3` for `Conv`, dropout rates etc.)
* __Zero overhead and [torchscript](https://pytorch.org/docs/stable/jit.html) support__

# Examples

For full functionality please check [__torchlayers documentation__](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge).
Below examples should introduce all necessary concepts you should know.

## Simple convolutional image and text classifier

* We will use single "model" for both tasks.
Firstly let's define it using `torch.nn` and `torchlayers`:

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

print(model)
```

Above would give you model's summary like this (__notice question marks for not yet inferred values__):

```python
Sequential(
  (0): Conv(in_channels=?, out_channels=64, kernel_size=3, stride=1, padding=same, dilation=1, groups=1, bias=True, padding_mode=zeros)
  (1): ReLU()
  (2): BatchNorm(num_features=?, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Conv(in_channels=?, out_channels=128, kernel_size=3, stride=1, padding=same, dilation=1, groups=1, bias=True, padding_mode=zeros)
  (4): ReLU()
  (5): Conv(in_channels=?, out_channels=256, kernel_size=11, stride=1, padding=same, dilation=1, groups=1, bias=True, padding_mode=zeros)
  (6): GlobalMaxPool()
  (7): Linear(in_features=?, out_features=10, bias=True)
)
```

* Now you can __build__/instantiate your model with example input (in this case MNIST-like):

```python
mnist_model = torchlayers.build(model, torch.randn(1, 3, 28, 28))
```

* Or if it's text classification you are after, same model could be built with different
`input shape` (e.g. for text classification using `300` dimensional pretrained embedding):

```python
# [batch, embedding, timesteps], first dimension > 1 for BatchNorm1d to work
text_model = torchlayers.build(model, torch.randn(2, 300, 1))
```

* Finally, you can `print` both models after instantiation, provided below side
by-side for readability (__notice different dimenstionality, e.g. `Conv2d` vs `Conv1d` after `torchlayers.build`__):

```python
                # MNIST CLASSIFIER                TEXT CLASSIFIER

                Sequential(                       Sequential(
                  (0): Conv1d(300, 64)              (0): Conv2d(3, 64)
                  (1): ReLU()                       (1): ReLU()
                  (2): BatchNorm1d(64)              (2): BatchNorm2d(64)
                  (3): Conv1d(64, 128)              (3): Conv2d(64, 128)
                  (4): ReLU()                       (4): ReLU()
                  (5): Conv1d(128, 256)             (5): Conv2d(128, 256)
                  (6): GlobalMaxPool()              (6): GlobalMaxPool()
                  (7): Linear(256, 10)              (7): Linear(256, 10)
                )                                 )
```

As you can see both modules "compiled" into original `pytorch` layers.

## Custom modules with shape inference capabilities

User can define any module and make it shape inferable with `torchlayers.Infer`
decorator class:

```python
@torchlayers.Infer()  # Remember to instantiate it
class MyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight, self.bias)


layer = MyLinear(out_features=32)
# [WIP] Currently custom layers are unbuildable, you can still use them without build though
layer(torch.randn(1, 64))
```

By default `inputs.shape[1]` will be used as `in_features` value
during initial `forward` pass. If you wish to use different `index` (e.g. to infer using
`inputs.shape[3]`) use `@torchlayers.Infer(index=3)` as a decorator.

## Autoencoder with inverted residual bottleneck and pixel shuffle

Please check code comments and [__documentation__](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)
if needed. If you are unsure what autoencoder is you could see
[__this example blog post__](https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726).

Below is a convolutional denoising autoencoder example for `ImageNet`-like images.
Think of it like a demonstration of capabilities of different layers
and building blocks provided by `torchlayers`.


```python
# Input - 3 x 256 x 256 for ImageNet reconstruction
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchlayers.Sequential(
            torchlayers.StandardNormalNoise(),  # Apply noise to input images
            torchlayers.Conv(64, kernel_size=7),
            torchlayers.activations.Swish(),  # Direct access to module .activations
            torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
            torchlayers.AvgPool(),  # shape 64 x 128 x 128, kernel_size=2 by default
            torchlayers.HardSwish(),  # Access simply through torchlayers
            torchlayers.SeparableConv(128),  # Up number of channels to 128
            torchlayers.InvertedResidualBottleneck(),  # Default with squeeze excitation
            torch.nn.ReLU(),
            torchlayers.AvgPool(),  # shape 128 x 64 x 64, kernel_size=2 by default
            torchlayers.DepthwiseConv(256),  # DepthwiseConv easier to use
            # Pass input thrice through the same weights like in PolyNet
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.ReLU(),  # all torch.nn can be accessed via torchlayers
            torchlayers.MaxPool(),  # shape 256 x 32 x 32
            torchlayers.Fire(out_channels=512),  # shape 512 x 32 x 32
            torchlayers.SqueezeExcitation(hidden=64),
            torchlayers.InvertedResidualBottleneck(),
            torchlayers.MaxPool(),  # shape 512 x 16 x 16
            torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
            # Randomly switch off the last two layers with 0.5 probability
            torchlayers.StochasticDepth(
                torch.nn.Sequential(
                    torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
                    torchlayers.InvertedResidualBottleneck(squeeze_excitation=False),
                ),
                p=0.5,
            ),
            torchlayers.AvgPool(),  # shape 512 x 8 x 8
        )

        # This one is more "standard"
        self.decoder = torchlayers.Sequential(
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=2),
            # Has ICNR initialization by default after calling `build`
            torchlayers.ConvPixelShuffle(out_channels=512, upscale_factor=2),
            # Shape 512 x 16 x 16 after PixelShuffle
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.ConvPixelShuffle(out_channels=256, upscale_factor=2),
            # Shape 256 x 32 x 32
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.ConvPixelShuffle(out_channels=128, upscale_factor=2),
            # Shape 128 x 64 x 64
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=4),
            torchlayers.ConvPixelShuffle(out_channels=64, upscale_factor=2),
            # Shape 64 x 128 x 128
            torchlayers.InvertedResidualBottleneck(),
            torchlayers.Conv(256),
            torchlayers.Dropout(),  # Defaults to 0.5 and Dropout2d for images
            torchlayers.Swish(),
            torchlayers.InstanceNorm(),
            torchlayers.ConvPixelShuffle(out_channels=32, upscale_factor=2),
            # Shape 32 x 256 x 256
            torchlayers.Conv(16),
            torchlayers.Swish(),
            torchlayers.Conv(3),
            # Shape 3 x 256 x 256
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))
```

Now one can instantiate the module and use it with `torch.nn.MSELoss` as per usual.

```python
autoencoder = torchlayers.build(AutoEncoder(), torch.randn(1, 3, 256, 256))
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

# Contributing

If you find issue or would like to see some functionality (or implement one), please [open new Issue](https://help.github.com/en/articles/creating-an-issue) or [create Pull Request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).
