# Third argument
convolution = ("SqueezeExcitation", "Fire", "Conv", "ConvTranspose", "InvertedResidual")

normalization = ("BatchNorm", "InstanceNorm", "GroupNorm")

upsample = ("ConvPixelShuffle",)


def all():
    return convolution + normalization + upsample
