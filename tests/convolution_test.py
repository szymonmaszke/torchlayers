import itertools

import torch

import pytest
import torchlayers


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
            torchlayers.Dropout(),  # Default 0.5 and Dropout2d for images
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

        # Will make this one easier and repetitive
        self.decoder = torchlayers.Sequential(
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=2),
            # Has ICNR initialization by default as well
            torchlayers.ConvPixelShuffle(out_channels=512, upscale_factor=2),
            # Shape 512 x 16 x 16 after PixelShuffle
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.ConvPixelShuffle(out_channels=256, upscale_factor=2),
            torchlayers.StandardNormalNoise(),  # add Gaussian Noise
            # Shape 256 x 32 x 32
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=3),
            torchlayers.ConvPixelShuffle(out_channels=128, upscale_factor=2),
            # Shape 128 x 64 x 64
            torchlayers.Poly(torchlayers.InvertedResidualBottleneck(), order=4),
            torchlayers.ConvPixelShuffle(out_channels=64, upscale_factor=2),
            # Shape 64 x 128 x 128
            torchlayers.InvertedResidualBottleneck(),
            torchlayers.Conv(256),
            torchlayers.Swish(),
            torchlayers.BatchNorm(),
            torchlayers.ConvPixelShuffle(out_channels=32, upscale_factor=2),
            # Shape 32 x 256 x 256
            torchlayers.Conv(16),
            torchlayers.Swish(),
            torchlayers.Conv(3),
            # Shape 3 x 256 x 256
        )

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


@pytest.fixture
def classification_model():
    return torchlayers.Sequential(
        torchlayers.Conv(64),
        torchlayers.ReLU(),
        torchlayers.MaxPool(),
        torchlayers.Residual(
            torchlayers.Sequential(
                torchlayers.Conv(64, groups=16),
                torchlayers.ReLU(),
                torchlayers.GroupNorm(num_groups=4),
                torchlayers.Conv(64, groups=16),
                torchlayers.ChannelShuffle(groups=16),
                torchlayers.ReLU(),
            )
        ),
        torchlayers.SqueezeExcitation(),
        torchlayers.Sequential(
            torchlayers.Dropout(),
            torchlayers.Conv(128),
            torchlayers.ReLU(),
            torchlayers.InstanceNorm(),
        ),
        torchlayers.Poly(
            torchlayers.WayPoly(
                torchlayers.Fire(128),
                torchlayers.Fire(128),
                torchlayers.Fire(128),
                torchlayers.Fire(128),
            ),
            order=2,
        ),
        torchlayers.AvgPool(),
        torchlayers.StochasticDepth(torchlayers.Fire(128, hidden_channels=64)),
        torchlayers.ReLU(),
        torchlayers.GlobalAvgPool(),
        torchlayers.Linear(64),
    )


@pytest.fixture
def autoencoder_model():
    return AutoEncoder()


def test_text_cnn():
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

    torchlayers.build(model, torch.randn(2, 300, 1))


def test_classification(classification_model):
    classification_model = torchlayers.build(
        classification_model, torch.randn(16, 3, 28, 28)
    )
    optimizer = torch.optim.Adam(classification_model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(16):
        output = classification_model(torch.randn(16, 3, 28, 28))
        loss = criterion(output, torch.randint(10, (16,)))
        loss.backward()

        optimizer.zero_grad()


def test_conv_pixel_shuffle():
    model = torchlayers.build(
        torchlayers.ConvPixelShuffle(out_channels=512, upscale_factor=2),
        torch.randn(1, 256, 128, 128),
    )


def test_residual_bottleneck():
    model = torchlayers.build(
        torchlayers.InvertedResidualBottleneck(), torch.randn(1, 128, 128, 128)
    )


def test_autoencoder(autoencoder_model):
    autoencoder_model = torchlayers.build(
        autoencoder_model, torch.randn(1, 3, 256, 256)
    )

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
        classification_model = torchlayers.build(
            torchlayers.Conv(
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
            classification_model = torchlayers.build(
                torchlayers.Conv(
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
