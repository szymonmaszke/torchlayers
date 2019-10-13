import typing

import torch

from ._dev_utils.infer import create_forward, create_repr

# Make inheritable torchlayers.Inferable like module base
# def infer(module_class: torch.nn.Module, input_name: str):
#     """Create your own inferable `module`.

#     First argument will be `inferred` from the first (including batch) dimension
#     of it's input.

#     Pass your own class of `module` and name of it's first input argument.

#     Parameters
#     ----------
#     module_class : torch.nn.Module
#             Class of module for which it's input will be inferred
#     input_name : str
#             Name of input (used in custom representation)

#     Returns
#     -------
#     Infer
#             Proxy class callable just like your custom module (except for the first argument).
#             Will infer first dimension for you.

#     """

#     class Infer(torch.nn.Module):
#         def __init__(self, *args, **kwargs):
#             self._inferred_module_class = module_class
#             for key, value in kwargs.items():
#                 setattr(self, key, value)

#             _non_inferable_parameters = (key for key in kwargs)
#             self._forward = create_forward(_non_inferable_parameters)
#             self._repr = create_repr(input_name)

#         def forward(self, inputs, *args, **kwargs):
#             self._forward(self, inputs, *args, **kwargs)

#         def __repr__(self):
#             return self._repr(self)

#     return Infer


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom functions within
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input.

    Parameters
    ----------
    function: typing.Callable
            Any argument user specified function

    Returns
    -------
    typing.Any
            Anything `function` returns

    """

    def __init__(self, function: typing.Callable):
        super().__init__()
        self.function: typing.Callable = function

    def forward(self, *args, **kwargs) -> typing.Any:
        return self.function(*args, **kwargs)


class Flatten(torch.nn.Module):
    """Flatten input (except batch dimension).

    Returns
    -------
    torch.Tensor
            Flattened input `torch.Tensor`.

    """

    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)
