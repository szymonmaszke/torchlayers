# First argument - batch
# Third argument to infer
recurrent = ("RNN", "LSTM", "GRU")


# Second argument to infer
recurrent_cells = ("RNNCell", "LSTMCell", "GRUCell")

convolution = (
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
)

normalization = (
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "SyncBatchNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
)

linear = ("Linear",)


attention = ("MultiheadAttention",)

transformer = ("Transformer", "TransformerEncoderLayer", "TransformerDecoderLayer")


def all():
    return (
        recurrent
        + recurrent_cells
        + convolution
        + normalization
        + linear
        + attention
        + transformer
    )
