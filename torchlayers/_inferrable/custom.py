# Third argument
convolution = (
    "SqueezeExcitation",
    "Fire",
    "Conv",
    "ConvTranspose",
    "InvertedResidualBottleneck",
)

normalization = ("BatchNorm", "InstanceNorm", "GroupNorm")

upsample = ("ConvPixelShuffle",)


def all():
    return convolution + normalization + upsample
