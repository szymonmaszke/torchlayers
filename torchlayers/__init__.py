import inspect

import torch

from . import (_dev_utils, _inferable, convolution, normalization, pooling,
               regularization)
from ._general import Flatten, Lambda
from ._version import __version__


def _getattr(name):
    module_class = None
    for module in (convolution, normalization, pooling, regularization, torch.nn):
        module_class = getattr(module, name, None)
        if module_class is not None:
            return module_class

    raise AttributeError(f"module {__name__} has no attribute {name}")


def make_inferrable(module_class):
    name = module_class.__name__
    inferred_module = type(
        name, (torch.nn.Module,), {_dev_utils.infer.MODULE_CLASS: module_class}
    )
    signature = inspect.signature(module_class.__init__)
    arguments = [str(argument) for argument in signature.parameters.values()]

    # Arguments: self, input, *
    setattr(inferred_module, "__init__", _dev_utils.infer.create_init(*arguments[2:]))
    setattr(
        inferred_module,
        "forward",
        _dev_utils.infer.create_forward(
            _dev_utils.infer.MODULE, _dev_utils.infer.MODULE_CLASS, *arguments[2:]
        ),
    )
    setattr(
        inferred_module,
        "__repr__",
        _dev_utils.infer.create_repr(_dev_utils.infer.MODULE, **{arguments[1]: "?"}),
    )
    setattr(
        inferred_module,
        "__getattr__",
        _dev_utils.infer.create_getattr(_dev_utils.infer.MODULE),
    )

    setattr(
        inferred_module,
        "__reduce__",
        _dev_utils.infer.create_reduce(_dev_utils.infer.MODULE, *arguments[1:]),
    )

    return inferred_module


###############################################################################
#
#                       MODULE ATTRIBUTE GETTERS
#
###############################################################################


# Fix dir
def __dir__():
    return dir(torch.nn)


def __getattr__(name: str):
    module_class = _getattr(name)
    if name in _inferable.torch.all() + _inferable.custom.all():
        return make_inferrable(module_class)
    return module_class
