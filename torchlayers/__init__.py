import inspect
import io
import types
import typing
import warnings

import torch

from . import (_dev_utils, activations, convolution, normalization, pooling,
               regularization, upsample)
from ._version import __version__
from .module import InferDimension

__all__ = ["build", "infer", "Lambda", "Reshape", "Concatenate"]


def build(module, *args, **kwargs):
    """Build PyTorch layer or module by providing example input.

    This method should be used **always** after creating module using `torchlayers`
    and shape inference especially.

    Works similarly to `build` functionality provided by `keras`.

    Provided module will be "compiled" to PyTorch primitives to remove any
    overhead.

    `torchlayers` also supports `post_build` function to perform some action after
    shape was inferred (weight initialization example below)::


        import torch
        import torchlayers as tl

        class _MyModuleImpl(torch.nn.Linear):
            def post_build(self):
                # You can do anything here really
                torch.nn.init.eye_(self.weights)

        MyModule = tl.infer(_MyModuleImpl)

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


def infer(module_class, index: str = 1):
    """Allows custom user modules to infer input shape.

    Input shape should be the first argument after `self`.

    Usually used as class decorator, e.g.::

        import torch
        import torchlayers as tl

        class _StrangeLinearImpl(torch.nn.Linear):
            def __init__(self, in_features, out_features, bias: bool = True):
                super().__init__(in_features, out_features, bias)
                self.params = torch.nn.Parameter(torch.randn(out_features))

            def forward(self, inputs):
                super().forward(inputs) + self.params

        # Now you can use shape inference of in_features
        StrangeLinear = tl.infer(_StrangeLinearImpl)

        # in_features can be inferred
        layer = StrangeLinear(out_features=64)


    Parameters
    ----------
    module_class: torch.nn.Module
        Class of module to be updated with shape inference capabilities.

    index: int, optional
        Index into `tensor.shape` input which should be inferred, e.g. tensor.shape[1].
        Default: `1` (`0` being batch dimension)

    """

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
            infered_module, "__init__", _dev_utils.infer.create_init(parsed_arguments),
        )

        setattr(
            infered_module,
            "forward",
            _dev_utils.infer.create_forward(
                _dev_utils.infer.MODULE,
                _dev_utils.infer.MODULE_CLASS,
                parsed_arguments,
                index,
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
            _dev_utils.infer.create_reduce(_dev_utils.infer.MODULE, parsed_arguments),
        )

        return infered_module

    return module_class


class Lambda(torch.nn.Module):
    """Use any function as `torch.nn.Module`

    Simple proxy which allows you to use your own custom in
    `torch.nn.Sequential` and other requiring `torch.nn.Module` as input::

        import torch
        import torchlayers as tl

        model = torch.nn.Sequential(tl.Lambda(lambda tensor: tensor ** 2))
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

        import torch
        import torchlayers as tl

        class Foo(torch.nn.Module):
            # Return same tensor three times
            # You could explicitly return a list or tuple as well
            def forward(tensor):
                return tensor, tensor, tensor


        model = torch.nn.Sequential(Foo(), tl.Concatenate())
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

        import torch
        import torchlayers as tl

        layer = tl.Reshape(20, -1)
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
#                       MODULE AND SHAPE INFERENCE
#
###############################################################################


modules_map = {
    # PyTorch specific tensor index
    "Linear": (torch.nn, -1),
    "RNN": (torch.nn, 2),
    "LSTM": (torch.nn, 2),
    "GRU": (torch.nn, 2),
    "MultiheadAttention": (torch.nn, 2),
    "Transformer": (torch.nn, 2),
    "TransformerEncoderLayer": (torch.nn, 2),
    "TransformerDecoderLayer": (torch.nn, 2),
    # PyTorch default (1) tensor index
    "RNNCell": torch.nn,
    "LSTMCell": torch.nn,
    "GRUCell": torch.nn,
    "Conv1d": torch.nn,
    "Conv2d": torch.nn,
    "Conv3d": torch.nn,
    "ConvTranspose1d": torch.nn,
    "ConvTranspose2d": torch.nn,
    "ConvTranspose3d": torch.nn,
    "BatchNorm1d": torch.nn,
    "BatchNorm2d": torch.nn,
    "BatchNorm3d": torch.nn,
    "SyncBatchNorm": torch.nn,
    "InstanceNorm1d": torch.nn,
    "InstanceNorm2d": torch.nn,
    "InstanceNorm3d": torch.nn,
    # Torchlayers convolution
    "SqueezeExcitation": convolution,
    "Fire": convolution,
    "Conv": convolution,
    "ConvTranspose": convolution,
    "DepthwiseConv": convolution,
    "SeparableConv": convolution,
    "InvertedResidualBottleneck": convolution,
    # Torchlayers normalization
    "BatchNorm": normalization,
    "InstanceNorm": normalization,
    "GroupNorm": normalization,
    # Torchlayers upsample
    "ConvPixelShuffle": upsample,
}


def __dir__():
    return [key for key in modules_map if not key.startswith("*")] + dir(torch.nn)


def __getattr__(name: str):
    def infer_module(module, tensor_index):
        klass = getattr(module, name, None)
        if klass is not None:
            return infer(klass, tensor_index)
        return None

    def process_entry(data) -> typing.Optional:
        if isinstance(data, typing.Iterable):
            if len(data) != 2:
                raise AttributeError(
                    f"Value of to-be-inferred module: {name} second argument has "
                    f"to be of length 2 but got {len(data)}. Check torchlayers documentation."
                )
            return infer_module(*data)

        if isinstance(data, types.ModuleType):
            return infer_module(data, 1)

        raise AttributeError(
            "Torchlayers recognized entry to infer but it's entry is of incorrect type. "
            "Check documentation about registration of user modules for more info."
        )

    def registered_module():
        for key, data in modules_map.items():
            if key.startswith("*"):
                klass = process_entry(data)
                if klass is not None:
                    return klass

        return None

    def noninferable_torchlayers():
        for module in (
            activations,
            convolution,
            normalization,
            pooling,
            regularization,
            upsample,
        ):
            klass = getattr(module, name, None)
            if klass is not None:
                return klass

        return None

    data = modules_map.get(name)
    if data is None:
        klass = registered_module()
    else:
        klass = process_entry(data)

    # Try to return from torchlayers without inference
    if klass is None:
        klass = noninferable_torchlayers()

    # Try to return from torch.nn without inference
    if klass is None:
        klass = getattr(torch.nn, name, None)

    # As far as we know there is no such module
    if klass is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    return klass
