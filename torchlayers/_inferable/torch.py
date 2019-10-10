# Third argument
recurrent = ("RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell")
# Second argument
convolution = (
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
)

normalization = (
    "BatchNorm1d",  # Remove all d from dimensions
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
)

linear = ("Linear",)


def all():
    return recurrent + convolution + normalization + linear
