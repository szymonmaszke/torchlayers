import typing

import torch


class Representation(torch.nn.Module):
    def __repr__(self):
        parameters = ", ".join(
            [
                "{}={}".format(key, value)
                for key, value in vars(self).items()
                if not key.startswith("_") and key != "training"
            ]
        )
        return "{}({})".format(type(self).__name__, parameters)
