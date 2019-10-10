# Third argument
convolution = ("SqueezeExcitation", "Fire", "Conv", "ConvTranspose")

normalization = ("BatchNorm", "InstanceNorm", "GroupNorm")


def all():
    return convolution + normalization
