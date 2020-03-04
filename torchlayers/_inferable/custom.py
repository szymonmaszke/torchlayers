# Third argument
convolution = (
    "SqueezeExcitation",
    "Fire",
    "Conv",
    "ConvTranspose",
    "DepthwiseConv",
    "SeparableConv",
    "InvertedResidualBottleneck",
)

normalization = ("BatchNorm", "InstanceNorm", "GroupNorm")

upsample = ("ConvPixelShuffle",)


def all():
    return convolution + normalization + upsample
