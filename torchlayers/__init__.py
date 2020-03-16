import inspect
import io
import typing
import warnings

import torch

from . import (_dev_utils, _inferable, activations, convolution, normalization,
               pooling, regularization, upsample)
from ._version import __version__

__all__ = ["build", "Infer", "Lambda", "Reshape", "Concatenate"]


def build(module, *args, **kwargs):
    """Build PyTorch layer or module by providing example input.

    This method should be used **always** after creating module using `torchlayers`
    and shape inference especially.

    Works similarly to `build` functionality provided by `keras`.

    Provided module will be "compiled" to PyTorch primitives to remove any
    overhead.

    `torchlayers` also supports `post_build` function to perform some action after
    shape was inferred (weight initialization example below)::


        @torchlayers.infer
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

    def torch_compile(module):
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
    module = torch_compile(module)
    run_post(module)
    return module


class Infer:
    """Allows custom user modules to infer input shape.

    Input shape should be the first argument after `self`.

    Usually used as class decorator, e.g.::

        # Remember it's a class, it has to be instantiated
        @torchlayers.Infer()
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
    index: int, optional
        Index into `tensor.shape` input which should be inferred, e.g. tensor.shape[1].
        Default: `1` (`0` being batch dimension)

    """

    def __init__(self, index: int = 1):
        self.index: int = index

    def __call__(self, module_class):
        init_arguments = [
            str(argument)
            for argument in inspect.signature(module_class.__init__).parameters.values()
        ]

        # Other argument than self
        if len(init_arguments) > 1:
            name = module_class.__name__
            infered_module = type(
                name, (torch.nn.Module,), {_dev_utils.infer.MODULE_CLASS: module_class},
            )
            parsed_arguments, uninferable_arguments = _dev_utils.infer.parse_arguments(
                init_arguments, infered_module
            )

            setattr(
                infered_module,
                "__init__",
                _dev_utils.infer.create_init(parsed_arguments),
            )

            setattr(
                infered_module,
                "forward",
                _dev_utils.infer.create_forward(
                    _dev_utils.infer.MODULE,
                    _dev_utils.infer.MODULE_CLASS,
                    parsed_arguments,
                    self.index,
                ),
            )
            setattr(
                infered_module,
                "__repr__",
                _dev_utils.infer.create_repr(
                    _dev_utils.infer.MODULE, **uninferable_arguments
                ),
            )
            setattr(
                infered_module,
                "__getattr__",
                _dev_utils.infer.create_getattr(_dev_utils.infer.MODULE),
            )

            setattr(
                infered_module,
                "__reduce__",
                _dev_utils.infer.create_reduce(
                    _dev_utils.infer.MODULE, parsed_arguments
                ),
            )

            return infered_module

        return module_class


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom in
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input::

        model = torch.nn.Sequential(torchlayers.Lambda(lambda tensor: tensor ** 2))
        model(torch.randn(64 , 20))

    Parameters
    ----------
    function : Callable
        Any user specified function

    Returns
    -------
    Any
        Anything `function` returns

    """

    def __init__(self, function: typing.Callable):
        super().__init__()
        self.function: typing.Callable = function

    def forward(self, *args, **kwargs) -> typing.Any:
        return self.function(*args, **kwargs)


class Concatenate(torch.nn.Module):
    """Concatenate list of tensors.

    Mainly useful in `torch.nn.Sequential` when previous layer returns multiple
    tensors, e.g.::

        class Foo(torch.nn.Module):
            # Return same tensor three times
            # You could explicitly return a list or tuple as well
            def forward(tensor):
                return tensor, tensor, tensor


        model = torch.nn.Sequential(Foo(), torchlayers.Concatenate())
        model(torch.randn(64 , 20))

    All tensors must have the same shape (except in the concatenating dimension).

    Parameters
    ----------
    dim : int
        Dimension along which tensors will be concatenated

    Returns
    -------
    torch.Tensor
        Concatenated tensor along specified `dim`.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Reshape(torch.nn.Module):
    """Reshape tensor excluding `batch` dimension

    Reshapes input `torch.Tensor` features while preserving batch dimension.
    Standard `torch.reshape` values (e.g. `-1`) are supported, e.g.::

        layer = torchlayers.Reshape(20, -1)
        layer(torch.randn(64, 80)) # shape (64, 20, 4)

    All tensors must have the same shape (except in the concatenating dimension).
    If possible, no copy of `tensor` will be performed.

    Parameters
    ----------
    shapes: *int
        Variable length list of shapes used in view function

    Returns
    -------
    torch.Tensor
        Concatenated tensor

    """

    def __init__(self, *shapes: int):
        super().__init__()
        self.shapes: typing.Tuple[int] = shapes

    def forward(self, tensor):
        return torch.reshape(tensor, (tensor.shape[0], *self.shapes))


###############################################################################
#
#                       MODULE ATTRIBUTE GETTERS
#
###############################################################################


def __dir__():
    return (
        dir(torch.nn)
        + ["Lambda", "Concatenate", "Reshape"]
        + dir(convolution)
        + dir(normalization)
        + dir(upsample)
        + dir(pooling)
        + dir(regularization)
        + dir(activations)
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
            activations,
            torch.nn,
        ):
            module_class = getattr(module, name, None)
            if module_class is not None:
                return module_class

        raise AttributeError("module {} has no attribute {}".format(__name__, name))

    module_class = _getattr(name)
    if name in _inferable.torch.all() + _inferable.custom.all():
        return Infer(_dev_utils.helpers.get_per_module_index(module_class))(
            module_class
        )
    return module_class
