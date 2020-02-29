import inspect
import io
import warnings

import torch

from . import (_dev_utils, _inferable, convolution, normalization, pooling,
               regularization, upsample)
from ._general import Flatten, Lambda
from ._version import __version__


def build(module, *args, **kwargs):
    """
    Build PyTorch layer or module by providing example input.

    This method should be used **always** after creating module using `torchlayers`
    and shape inference especially.

    Works similarly to `build` functionality provided by `keras`.

    Provided module will be "compiled" to PyTorch primitives to remove any
    overhead.

    Parameters
    ----------
    module : torch.nn.Module
        Instance of module to build.
    *args
        Arguments required by module
    **kwargs
        Keyword arguments
    """

    def cast_to_torch(module):
        with io.BytesIO() as buffer:
            torch.save(module, buffer)
            return torch.load(io.BytesIO(buffer.getvalue()))

    def run_post(module):
        for submodule in module.modules():
            function = getattr(submodule, "post_build", None)
            if function is not None:
                post_build = getattr(submodule, "post_build")
                if not callable(post_build):
                    raise ValueError(
                        "{submodule}'s post_build is required to be a method."
                    )
                submodule.post_build()

    with torch.no_grad():
        module.eval()
        module(*args, **kwargs)
    module.train()
    module = cast_to_torch(module)
    run_post(module)
    return module


def make_inferrable(module_class):
    name = module_class.__name__
    inferred_module = type(
        name, (torch.nn.Module,), {_dev_utils.infer.MODULE_CLASS: module_class}
    )

    init_signature = inspect.signature(module_class.__init__)
    init_arguments = [str(argument) for argument in init_signature.parameters.values()]

    setattr(
        inferred_module, "__init__", _dev_utils.infer.create_init(init_arguments[2:])
    )
    setattr(
        inferred_module,
        "forward",
        _dev_utils.infer.create_forward(
            _dev_utils.infer.MODULE, _dev_utils.infer.MODULE_CLASS, init_arguments[2:]
        ),
    )
    setattr(
        inferred_module,
        "__repr__",
        _dev_utils.infer.create_repr(
            _dev_utils.infer.MODULE, **{init_arguments[1]: "?"}
        ),
    )
    setattr(
        inferred_module,
        "__getattr__",
        _dev_utils.infer.create_getattr(_dev_utils.infer.MODULE),
    )

    setattr(
        inferred_module,
        "__reduce__",
        _dev_utils.infer.create_reduce(_dev_utils.infer.MODULE, init_arguments[1:]),
    )

    return inferred_module


###############################################################################
#
#                       MODULE ATTRIBUTE GETTERS
#
###############################################################################


def __dir__():
    return (
        dir(torch.nn)
        + ["Flatten", "Lambda"]
        + dir(convolution)
        + dir(normalization)
        + dir(upsample)
        + dir(pooling)
        + dir(regularization)
    )


def __getattr__(name: str):
    def _getattr(name):
        module_class = None
        for module in (
            convolution,
            normalization,
            pooling,
            regularization,
            upsample,
            torch.nn,
        ):
            module_class = getattr(module, name, None)
            if module_class is not None:
                return module_class

        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_class = _getattr(name)
    if name in _inferable.torch.all() + _inferable.custom.all():
        return make_inferrable(module_class)
    return module_class
