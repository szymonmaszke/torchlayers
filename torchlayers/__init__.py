import collections
import inspect
import io
import types
import typing
import warnings

import torch

from . import (_dev_utils, activations, convolution, normalization, pooling,
               preprocessing, regularization, upsample)
from ._version import __version__
from .module import InferDimension

__all__ = ["summary", "build", "infer", "Lambda", "Reshape", "Concatenate"]


def summary(module, *args, **kwargs):
    """Create summary of PyTorch module.

    Works similarly to `summary` functionality provided by `keras`.

    Example::

        import torchlayers as tl


        class Classifier(tl.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = tl.Conv2d(64, kernel_size=6)
                self.conv2 = tl.Conv2d(128, kernel_size=3)
                self.conv3 = tl.Conv2d(256, kernel_size=3, padding=1)
                self.pooling = tl.GlobalMaxPool()
                self.dense = tl.Linear(10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                return self.dense(self.pooling(x))

        clf = Classifier()
        tl.build(clf, torch.randn(1, 3, 32, 32))
        # You have to print the summary explicitly
        print(tl.summary(clf, torch.randn(1, 3, 32, 32)))

    Above would print (please notice explicit usage of `print` statement)::


                Layer (type)       |      Inputs      |     Outputs      | Params (size in MB)  | Buffers (size in MB)
        ===============================================================================================================
        ---------------------------------------------------------------------------------------------------------------
               conv1 (Conv2d)      |  (1, 3, 32, 32)  | (1, 64, 27, 27)  |  6976 (0.027904 MB)  |   0 (0.000000 MB)
        ---------------------------------------------------------------------------------------------------------------
               conv2 (Conv2d)      | (1, 64, 27, 27)  | (1, 128, 25, 25) | 73856 (0.295424 MB)  |   0 (0.000000 MB)
        ---------------------------------------------------------------------------------------------------------------
               conv3 (Conv2d)      | (1, 128, 25, 25) | (1, 256, 25, 25) | 295168 (1.180672 MB) |   0 (0.000000 MB)
        ---------------------------------------------------------------------------------------------------------------
         pooling (GlobalMaxPool2d) | (1, 256, 25, 25) |     (1, 256)     |   0 (0.000000 MB)    |   0 (0.000000 MB)
        ---------------------------------------------------------------------------------------------------------------
               dense (Linear)      |     (1, 256)     |     (1, 10)      |  2570 (0.010280 MB)  |   0 (0.000000 MB)
        ---------------------------------------------------------------------------------------------------------------
        ===============================================================================================================
        Total params: 378570 (1.514280 Mb)
        Total buffers: 0 (0.000000 Mb)


    Custom `Summarizer` object is returned from the function. Users can specify which columns to return
    (if any) using `string` function.
    Calling `summary` on the above model like this::

        print(tl.summary(clf, torch.randn(1, 3, 32, 32)).string(buffers=False, inputs=False))

    Would give the following output to `stdout`:


                Layer (type)       |     Outputs      | Params (size in MB)
        =====================================================================
        ---------------------------------------------------------------------
               conv1 (Conv2d)      | (1, 64, 27, 27)  |  6976 (0.027904 MB)
        ---------------------------------------------------------------------
               conv2 (Conv2d)      | (1, 128, 25, 25) | 73856 (0.295424 MB)
        ---------------------------------------------------------------------
               conv3 (Conv2d)      | (1, 256, 25, 25) | 295168 (1.180672 MB)
        ---------------------------------------------------------------------
         pooling (GlobalMaxPool2d) |     (1, 256)     |   0 (0.000000 MB)
        ---------------------------------------------------------------------
               dense (Linear)      |     (1, 10)      |  2570 (0.010280 MB)
        ---------------------------------------------------------------------
        =====================================================================
        Total params: 378570 (1.514280 Mb)
        Total buffers: 0 (0.000000 Mb)

    `string` method has following flags (if `True` the column is returned, `True` by default)
    specifiable by user:

        - layers
        - inputs
        - outputs
        - params
        - buffers

    Returned `Summarizer` object has following fields which can be accessed and
    read/modified by user:

        - names - `list` containing names of consecutive modules
        - modules - `list` containing names of **classes** of consecutive modules
        - inputs - `list` containing inputs (possibly multiple of them) to layers.
        Each `input` is described by `shape` (if it's a `torch.Tensor` instance)
        or by class name otherwise
        - outputs - `list` containing outputs (possibly multiple of them) of layers.
        Each `output` is described by `shape` (if it's a `torch.Tensor` instance)
        or by class name otherwise
        - params - `list` containing number of parameters for each layer
        - params_sizes - `list` containing sizes of parameters for layers (in megabytes)
        - buffers - `list` containing number of buffer parameters for each layer
        - buffers_sizes - `list` containing sizes of buffers for layers (in megabytes)

    Finally, `Summarizer` can also be called with other inputs so it gathers
    representation based on the last provided input. For example::

        summarizer = tl.summary(module, torch.randn(1, 3, 32, 32))
        with summarizer:
            summarizer(torch.randn(5, 3, 64, 64))

        # Inputs and outputs summary for input of shape (5, 3, 64, 64)
        print(summarizer)

    .. note::
            This function can be called before or after `torchlayers.build`
            but the results may vary (names and types of layers before and after
            inference).

    .. note::
            This function runs input through the network hence the input shapes
            are inferred after this call. Network will not be built into
            PyTorch counterparts though (e.g. `tl.Conv` will not become `torch.nn.Conv2d` or a-like).


    Parameters
    ----------
    module : torch.nn.Module
        Instance of module to create summary for
    *args
        Arguments required by module's `forward`
    **kwargs
        Keyword arguments required by module's `forward`
    """

    class Summarizer:
        def __init__(self, module):
            self.module = module

            self.names = []
            self.modules = []

            self.inputs = []
            self.outputs = []

            self.params = []
            self.params_sizes = []

            self.buffers = []
            self.buffers_sizes = []

            self._handles = []
            self._last_module_state = None

        def __enter__(self):
            self._last_module_state = self.module.training
            self.module.eval()
            for name, submodule in list(self.module.named_modules())[1:]:
                if "." not in name:
                    self.names.append(name)
                    self.modules.append(type(submodule).__name__)
                    self._handles.append(submodule.register_forward_hook(self.hook))
            return self

        def __exit__(self, *_, **__):
            for handle in self._handles:
                handle.remove()
            self.module.train(self._last_module_state)
            return False

        @staticmethod
        def _parse(item):
            if torch.is_tensor(item):
                return tuple(item.shape)

            if isinstance(item, collections.abc.Iterable):
                return Summarizer._unwrap(
                    tuple(Summarizer._parse(element) for element in item)
                )
            else:
                return type(item).__name__

        @staticmethod
        def _unwrap(item):
            if len(item) == 1:
                return item[0]
            return item

        def hook(self, module, inputs, outputs):
            def _elements_count(iterator):
                return sum(element.numel() for element in iterator)

            def _elements_size_in_mb(iterator):
                return (
                    sum(
                        element.numel() * element.element_size() for element in iterator
                    )
                    / 1_000_000
                )

            # Inputs and outputs
            self.inputs.append(Summarizer._parse(inputs))
            self.outputs.append(Summarizer._parse(outputs))

            # Parameters and buffers
            self.params.append(_elements_count(module.parameters()))
            self.params_sizes.append(_elements_size_in_mb(module.parameters()))
            self.buffers.append(_elements_count(module.buffers()))
            self.buffers_sizes.append(_elements_size_in_mb(module.buffers()))

        def __call__(self, *args, **kwargs):
            with torch.no_grad():
                return self.module(*args, **kwargs)
            return self

        def string(
            self,
            layers: bool = True,
            inputs: bool = True,
            outputs: bool = True,
            params: bool = True,
            buffers: bool = True,
        ):
            def create_text():
                def conditional_return(conditions, values):
                    for condition, value in zip(conditions, values):
                        if condition:
                            yield value

                def footer(
                    total_params, total_params_sizes, total_buffers, total_buffers_sizes
                ):
                    return "Total params: {} ({:.6f} MB)\nTotal buffers: {} ({:.6f} MB)".format(
                        total_params,
                        total_params_sizes,
                        total_buffers,
                        total_buffers_sizes,
                    )

                layers_text = ["Layer (type)"]
                inputs_text = ["Inputs"]
                outputs_text = ["Outputs"]
                params_text = ["Params (size in MB)"]
                buffers_text = ["Buffers (size in MB)"]

                total_params, total_params_sizes, total_buffers, total_buffers_sizes = (
                    0,
                    0,
                    0,
                    0,
                )
                for (
                    name,
                    module,
                    inp,
                    out,
                    param,
                    param_size,
                    buffer,
                    buffer_size,
                ) in zip(
                    self.names,
                    self.modules,
                    self.inputs,
                    self.outputs,
                    self.params,
                    self.params_sizes,
                    self.buffers,
                    self.buffers_sizes,
                ):

                    layers_text.append("{} ({})".format(name, module))
                    inputs_text.append("{}".format(inp))
                    outputs_text.append("{}".format(out))
                    params_text.append("{} ({:.6f} MB)".format(param, param_size))
                    buffers_text.append("{} ({:.6f} MB)".format(buffer, buffer_size))
                    total_params += param
                    total_params_sizes += param_size
                    total_buffers += buffer
                    total_buffers_sizes += buffer_size

                return (
                    footer(
                        total_params,
                        total_params_sizes,
                        total_buffers,
                        total_buffers_sizes,
                    ),
                    conditional_return(
                        (layers, inputs, outputs, params, buffers),
                        (
                            layers_text,
                            inputs_text,
                            outputs_text,
                            params_text,
                            buffers_text,
                        ),
                    ),
                )

            def center(columns):
                # + 2 to leave some space for representation on both sides
                for column in columns:
                    longest = max(map(len, column)) + 2
                    yield tuple(map(lambda string: string.center(longest), column))

            def join(columns):
                return tuple("|".join(row) for row in zip(*columns))

            footer, columns = create_text()
            rows = join(center(columns))
            representation = rows[0]
            representation += "\n" + "=" * len(rows[0]) + "\n"
            representation += "-" * len(rows[0]) + "\n"
            for row in rows[1:]:
                representation += row
                representation += "\n" + "-" * len(row) + "\n"

            representation += "=" * len(rows[0]) + "\n"
            representation += footer
            return representation

        def __str__(self):
            return self.string()

    with Summarizer(module) as summarizer:
        summarizer(*args, **kwargs)
        return summarizer


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
            preprocessing,
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
