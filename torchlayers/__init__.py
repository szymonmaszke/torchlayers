import inspect
import io
import warnings

import torch

from . import (_dev_utils, _inferrable, activations, convolution,
               normalization, pooling, regularization, upsample)
from ._general import Flatten, Lambda
from ._version import __version__


def build(module, *args, **kwargs):
    """Build PyTorch layer or module by providing example input.

    This method should be used **always** after creating module using `torchlayers`
    and shape inference especially.

    Works similarly to `build` functionality provided by `keras`.

    Provided module will be "compiled" to PyTorch primitives to remove any
    overhead.

    `torchlayers` also supports `post_build` function to perform some action after
    shape was inferred (weight initialization example below):


        @torchlayers.inferrable
        class MyModule(torch.nn.Linear):
            def post_build(self):
                # You can do anything here really
                torch.nn.init.eye_(self.weights)

    `post_build` should have no arguments other than `self` so all necessary
    data should be saved in `module` beforehand.

    Parameters
    ----------
    module : torch.nn.Module
        Instance of module to build
    *args
        Arguments required by module's `forward`
    **kwargs
        Keyword arguments required by module's `forward`
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
                        "{}'s post_build is required to be a method.".format(submodule)
                    )
                submodule.post_build()

    with torch.no_grad():
        module.eval()
        module(*args, **kwargs)
    module.train()
    module = cast_to_torch(module)
    run_post(module)
    return module


def inferrable(module_class):
    """
    Allows custom user modules to infer input shape.

    Input shape should be the first argument after `self`.

    Usually used as class decorator, e.g.::

        @torchlayers.inferrable
        class StrangeLinear(torch.nn.Linear):
            def __init__(self, in_features, out_features, bias: bool = True):
                super().__init__(in_features, out_features, bias)
                self.params = torch.nn.Parameter(torch.randn(out_features))

            def forward(self, inputs):
                super().forward(inputs) + self.params

        # in_features can be inferred
        layer = StrangeLinear(out_features=64)


    Parameters
    ----------
    module_class : torch.nn.Module
        Module (layer-like) to give inferrable possibilities

    """
    name = module_class.__name__
    inferred_module = type(
        name, (torch.nn.Module,), {_dev_utils.infer.MODULE_CLASS: module_class}
    )

    init_signature = inspect.signature(module_class.__init__)
    init_arguments = [str(argument) for argument in init_signature.parameters.values()]

    # Only self, do not use inference if not required
    if len(init_arguments) > 1:
        parsed_arguments, uninferrable_arguments = _dev_utils.infer.parse_arguments(
            init_arguments, inferred_module
        )

        setattr(
            inferred_module, "__init__", _dev_utils.infer.create_init(parsed_arguments),
        )

        setattr(
            inferred_module,
            "forward",
            _dev_utils.infer.create_forward(
                _dev_utils.infer.MODULE,
                _dev_utils.infer.MODULE_CLASS,
                parsed_arguments,
            ),
        )
        setattr(
            inferred_module,
            "__repr__",
            _dev_utils.infer.create_repr(
                _dev_utils.infer.MODULE, **uninferrable_arguments
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
            _dev_utils.infer.create_reduce(_dev_utils.infer.MODULE, parsed_arguments),
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

        raise AttributeError("module {} has no attribute {}".format(__name__, name))

    module_class = _getattr(name)
    if name in _inferrable.torch.all() + _inferrable.custom.all():
        return inferrable(module_class)
    return module_class
