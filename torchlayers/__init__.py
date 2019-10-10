import collections
import functools
import inspect
import sys

import torch

from . import (_dev_utils, _inferable, convolution, normalization, pooling,
               regularization)
from ._general import Flatten, Lambda, infer
from ._version import __version__
from .convolution import *  # noqa
from .regularization import *  # noqa


def _getattr(name):
    module_class = None
    for module in (convolution, normalization, pooling, torch.nn):
        module_class = getattr(module, name, None)
        if module_class is not None:
            return module_class

    raise AttributeError(f"module {__name__} has no attribute {name}")


def _setup(name, module_class):
    inferred_module = type(
        name, (torch.nn.Module,), {"_inferred_module_class": module_class}
    )
    signature = inspect.signature(module_class.__init__)
    arguments = [str(argument) for argument in signature.parameters.values()]

    setattr(inferred_module, "__init__", _dev_utils.infer.create_init(arguments[2:]))
    setattr(inferred_module, "forward", _dev_utils.infer.create_forward(arguments[2:]))
    setattr(inferred_module, "__repr__", _dev_utils.infer.create_repr(arguments[1]))

    return inferred_module


def __dir__():
    return dir(torch.nn)


def __getattr__(name: str):
    module_class = _getattr(name)
    if name in _inferable.torch.all() + _inferable.custom.all():
        return _setup(name, module_class)
    return module_class
