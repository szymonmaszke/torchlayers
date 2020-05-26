import itertools

import torch

import pytest
import torchlayers as tl


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = tl.Sequential(
            tl.Conv(64, kernel_size=7),
            tl.activations.Swish(),  # Direct access to module .activations
            tl.InvertedResidualBottleneck(squeeze_excitation=False),
            tl.AvgPool(),  # shape 64 x 128 x 128, kernel_size=2 by default
            tl.HardSwish(),  # Access simply through tl
            tl.SeparableConv(128),  # Up number of channels to 128
            tl.InvertedResidualBottleneck(),  # Default with squeeze excitation
            torch.nn.ReLU(),
            tl.AvgPool(),  # shape 128 x 64 x 64, kernel_size=2 by default
            tl.DepthwiseConv(256),  # DepthwiseConv easier to use
            # Pass input thrice through the same weights like in PolyNet
            tl.Poly(tl.InvertedResidualBottleneck(), order=3),
            tl.ReLU(),  # all torch.nn can be accessed via tl
            tl.MaxPool(),  # shape 256 x 32 x 32
            tl.Fire(out_channels=512),  # shape 512 x 32 x 32
            tl.SqueezeExcitation(hidden=64),
            tl.InvertedResidualBottleneck(),
            tl.MaxPool(),  # shape 512 x 16 x 16
            tl.InvertedResidualBottleneck(squeeze_excitation=False),
            tl.Dropout(),  # Default 0.5 and Dropout2d for images
            # Randomly Switch the last two layers with 0.5 probability
            tl.StochasticDepth(
                torch.nn.Sequential(
                    tl.InvertedResidualBottleneck(squeeze_excitation=False),
                    tl.InvertedResidualBottleneck(squeeze_excitation=False),
                ),
                p=0.5,
            ),
            tl.AvgPool(),  # shape 512 x 8 x 8
        )

        # Will make this one easier and repetitive
        self.decoder = tl.Sequential(
            tl.Poly(tl.InvertedResidualBottleneck(), order=2),
            # Has ICNR initialization by default as well
            tl.ConvPixelShuffle(out_channels=512, upscale_factor=2),
            # Shape 512 x 16 x 16 after PixelShuffle
            tl.Poly(tl.InvertedResidualBottleneck(), order=3),
            tl.ConvPixelShuffle(out_channels=256, upscale_factor=2),
            tl.StandardNormalNoise(),  # add Gaussian Noise
            # Shape 256 x 32 x 32
            tl.Poly(tl.InvertedResidualBottleneck(), order=3),
            tl.ConvPixelShuffle(out_channels=128, upscale_factor=2),
            # Shape 128 x 64 x 64
            tl.Poly(tl.InvertedResidualBottleneck(), order=4),
            tl.ConvPixelShuffle(out_channels=64, upscale_factor=2),
            # Shape 64 x 128 x 128
            tl.InvertedResidualBottleneck(),
            tl.Conv(256),
            tl.Swish(),
            tl.BatchNorm(),
            tl.ConvPixelShuffle(out_channels=32, upscale_factor=2),
            # Shape 32 x 256 x 256
            tl.Conv(16),
            tl.Swish(),
            tl.Conv(3),
            # Shape 3 x 256 x 256
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


@pytest.fixture
def classification_model():
    return tl.Sequential(
        tl.Conv(64),
        tl.ReLU(),
        tl.MaxPool(),
        tl.Residual(
            tl.Sequential(
                tl.Conv(64, groups=16),
                tl.ReLU(),
                tl.GroupNorm(num_groups=4),
                tl.Conv(64, groups=16),
                tl.ChannelShuffle(groups=16),
                tl.ReLU(),
            )
        ),
        tl.SqueezeExcitation(),
        tl.Sequential(tl.Dropout(), tl.Conv(128), tl.ReLU(), tl.InstanceNorm(),),
        tl.Poly(
            tl.WayPoly(tl.Fire(128), tl.Fire(128), tl.Fire(128), tl.Fire(128),),
            order=2,
        ),
        tl.AvgPool(),
        tl.StochasticDepth(tl.Fire(128, hidden_channels=64)),
        tl.ReLU(),
        tl.GlobalAvgPool(),
        tl.Linear(64),
    )


@pytest.fixture
def autoencoder_model():
    return AutoEncoder()


def test_text_cnn():
    model = torch.nn.Sequential(
        tl.Conv(64),  # specify ONLY out_channels
        torch.nn.ReLU(),  # use torch.nn wherever you wish
        tl.BatchNorm(),  # BatchNormNd inferred from input
        tl.Conv(128),  # Default kernel_size equal to 3
        tl.ReLU(),
        tl.Conv(256, kernel_size=11),  # "same" padding as default
        tl.GlobalMaxPool(),  # Known from Keras
        tl.Linear(10),  # Output for 10 classes
    )

    tl.build(model, torch.randn(2, 300, 1))


def test_classification(classification_model):
    classification_model = tl.build(classification_model, torch.randn(16, 3, 28, 28))
    optimizer = torch.optim.Adam(classification_model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(16):
        output = classification_model(torch.randn(16, 3, 28, 28))
        loss = criterion(output, torch.randint(10, (16,)))
        loss.backward()

        optimizer.zero_grad()


def test_conv_pixel_shuffle():
    model = tl.build(
        tl.ConvPixelShuffle(out_channels=512, upscale_factor=2),
        torch.randn(1, 256, 128, 128),
    )


def test_residual_bottleneck():
    model = tl.build(tl.InvertedResidualBottleneck(), torch.randn(1, 128, 128, 128))


def test_autoencoder(autoencoder_model):
    autoencoder_model = tl.build(autoencoder_model, torch.randn(1, 3, 256, 256))

    optimizer = torch.optim.Adam(autoencoder_model.parameters())
    criterion = torch.nn.MSELoss()

    for _ in range(16):
        inputs = torch.randn(1, 3, 256, 256)
        output = autoencoder_model(inputs)
        loss = criterion(output, inputs)
        loss.backward()

        optimizer.zero_grad()


def test_same_padding_1d():
    inputs = torch.randn(1, 5, 125)
    for kernel_size, stride, dilation in itertools.product(
        range(1, 20, 2), *[range(1, 20, 2) for _ in range(2)]
    ):
        classification_model = tl.build(
            tl.Conv(
                5,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding="same",
            ),
            inputs,
        )
        output = classification_model(inputs)
        assert output.shape == inputs.shape


def test_same_padding_2d():
    inputs = torch.randn(1, 5, 100, 100)
    for kernel_size, stride, dilation in itertools.product(
        range(1, 15, 2), *[range(1, 8, 2) for _ in range(2)]
    ):
        for kernel_size2, stride2, dilation2 in itertools.product(
            range(1, 8, 2), *[range(1, 8, 2) for _ in range(2)]
        ):
            classification_model = tl.build(
                tl.Conv(
                    5,
                    kernel_size=(kernel_size, kernel_size2),
                    stride=(stride, stride2),
                    dilation=(dilation, dilation2),
                    padding="same",
                ),
                inputs,
            )
            output = classification_model(inputs)
            assert output.shape == inputs.shape
