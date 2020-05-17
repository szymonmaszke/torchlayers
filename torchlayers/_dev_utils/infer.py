import inspect
import pickle
import typing

from . import helpers

VARARGS_VARIABLE = "_torchlayers_varargs_variable"
KWARGS_VARIABLE = "_torchlayers_kwargs_variable"

MODULE = "_torchlayers_infered_module"
MODULE_CLASS = "_torchlayers_infered_class_module"


def parse_arguments(
    init_arguments: typing.List[str], module
) -> typing.Tuple[typing.List[str], typing.Dict[str, typing.Any]]:
    """Parse init arguments.

    This function will:

        - create list of names which cannot be determined from non-instantiated module
        (e.g. `input_shape`) and assign them values (later used for non-instantiated __repr__)
        - remove type hints from argument to be inferred (the first one after `self`)
        and default value if provided
        - remove type hints from all `__init__` arguments and preserve their default
        arguments (custom exec won't be able to parse type hints without explicit imports)
        - remove `self` so it won't be later assigned to dynamically created non-instantiated
        module
        - WORKAROUND: add arguments related to recurrent neural networks to the uninferable group as those
        only have *args and **kwargs and are otherwise unparsable.


    Parameters
    ----------
    init_arguments : List[str]
        __init__ arguments gathered by `inspect.signature(cls).parameters`
    module : type
        `type` metaclass inheriting from `torch.nn.Module` and named like original
        class (without shape inference)

    Returns
    -------
    Tuple[List[str], Dict[str, Any]]
        First item are arguments used for dynamic __init__ creation of inferable
        module. Second item is dictionary where key is name of argument and value can be anything.
        Those are uninferable arguments (not present in inferable __init__)
        and used solely for inferable module's __repr__ creation.

    """

    def _add_rnn_uninferable(uninferable: typing.Dict, module) -> typing.Dict:
        if module.__name__ in ["RNN", "LSTM", "GRU"]:
            uninferable.update(
                {
                    "input_size": "?",
                    "num_layers": 1,
                    "bias": False,
                    "batch_first": False,
                    "dropout": 0.0,
                    "bidirectional": False,
                }
            )
        return uninferable

    def _add_infered_shape_variable_name(init_arguments: str) -> typing.Dict:
        first_argument = helpers.remove_right_side(
            helpers.remove_type_hint(init_arguments[1])
        )
        if not helpers.is_vararg(first_argument) and not helpers.is_kwarg(
            first_argument
        ):
            # Remove the argument to be inferred
            init_arguments.pop(1)
            return {first_argument: "?"}

        return {}

    uninferable_arguments = _add_rnn_uninferable(
        _add_infered_shape_variable_name(init_arguments), module
    )

    # Pop self from arguments always so it won't be class-assigned later
    init_arguments.pop(0)
    return (
        [helpers.remove_type_hint(argument) for argument in init_arguments],
        uninferable_arguments,
    )


def create_init(parsed_init_arguments) -> typing.Callable:
    """
    Dynamically create inferable module's `__init__`.

    Function has to be executed in `namespace` dictionary in order
    to build it from parsed strings.

    Function is string representation of constructor with proper inheritance
    of `torch.nn.Module`.

    If `*args` or `**kwargs` are encountered in original `__init__` they will
    be saved for later  in `VARARGS_VARIABLE` and `KWARGS_VARIABLE` respectively.

    They will be unpacked after all arguments during first `forward` call when
    module is instantiated.

    Parameters
    ----------
    parsed_init_arguments : List[str]
        __init__ arguments parsed by `parse_arguments` function.

    Returns
    -------
    Callable
        Function being __init__ of uninstantiated module.

    """
    namespace = {}

    joined_arguments = ", ".join(parsed_init_arguments)
    function = """def __init__(self, {}):""".format(joined_arguments)
    function += "\n super(type(self), self).__init__()\n"

    for argument in parsed_init_arguments:
        no_defaults = helpers.remove_right_side(argument)

        # Is vararg save for forward use
        if helpers.is_vararg(no_defaults):
            function += " self.{} = {}\n".format(
                VARARGS_VARIABLE, helpers.remove_vararg(no_defaults)
            )
        # Same for kwawrg
        elif helpers.is_kwarg(no_defaults):
            function += " self.{} = {}\n".format(
                KWARGS_VARIABLE, helpers.remove_kwarg(no_defaults)
            )
        # "Normal" variable, assign to self
        else:
            function += " self.{0} = {0}\n".format(no_defaults)

    exec(function, namespace)
    return namespace["__init__"]


def create_forward(
    module, module_class, parsed_init_arguments, inference_index: int
) -> typing.Callable:
    """
    Return forward function which instantiates module after first pass.

    Based on module type either `input.shape[1]` or `input.shape[2]` (for RNNs)
    will be gathered from input and used as first argument during `__init__`
    call of instantiated module.
    After that arguments saved in non-instantiated module will be passed, followed
    by `varargs` and `kwargs` (if any).

    Module will be fully usable after the first pass (has parameters, can be trained)
    but it is highly advised to use `torchlayers.build` to remove any non-instantiated
    module overlay.


    Parameters
    ----------
    module : str
        Name of variable where instantiated module will be saved. Usually equal to
        global variable `MODULE`
    module_class : torch.nn.Module class
        Name of variable where `__class__` of module to be instantiated is kept.
        Used to instantiate module only.
    parsed_init_arguments : List[str]
        __init__ arguments parsed by `parse_arguments` function. Used to get names
        of variables saved in non-instantiated module to be passed to `__init__`
        of to be instantiated module.

    Returns
    -------
    Callable
        Function being `forward` of uninstantiated module.

    """

    def _non_vararg_kwarg(arguments):
        return (
            helpers.remove_right_side(argument)
            for argument in arguments
            if not helpers.is_vararg(argument) and not helpers.is_kwarg(argument)
        )

    def forward(self, inputs, *args, **kwargs):
        infered_module = getattr(self, module, None)
        if infered_module is None:
            module_cls = getattr(self, module_class)
            # All arguments from non-instantiated module
            init_args = [
                getattr(self, argument)
                for argument in _non_vararg_kwarg(parsed_init_arguments)
                # List of varargs (if any)
            ] + list(getattr(self, VARARGS_VARIABLE, ()))
            # Kwargs to be unpacked (if any)
            init_kwargs = getattr(self, KWARGS_VARIABLE, {})

            self.add_module(
                module,
                module_cls(
                    # Shape to be inferred. Either 1 for all modules or 2 for RNNs
                    inputs.shape[inference_index],
                    *init_args,
                    **init_kwargs
                ),
            )

            infered_module = getattr(self, module)

        return infered_module(inputs, *args, **kwargs)

    return forward


def create_repr(module, **non_inferable_names) -> typing.Callable:
    """
    Uninstantiated module representation.

    Representation is gathered from saved variables names and values of module
    to be instantiated, saved `*args` variables (if any) and saved `**kwargs` variables.

    Additionally names not saved in module (e.g. `input_shape` to be inferred)
    are added at the beginning. Usually their value is "?", RNNs being and edge
    case with all their arguments in `non_inferable_names` group
    (and except the ones provided explicitly by user).


    Parameters
    ----------
    module : str
        Name of variable where instantiated module will be saved. Usually equal to
        global variable `MODULE`
    **non_inferable_names : Dict[str, Any]
        Non-inferable names and their respective values of the module

    Returns
    -------
    Callable
        Function being `__repr__` of uninstantiated module.

    """

    def __repr__(self) -> str:
        infered_module = getattr(self, module, None)
        if infered_module is None:
            parameters = ", ".join(
                argument_representation
                for argument_representation in helpers.create_vars(
                    self, non_inferable_names, VARARGS_VARIABLE, KWARGS_VARIABLE
                )
            )
            return "{}({})".format(type(self).__name__, parameters)

        return repr(infered_module)

    return __repr__


def create_getattr(module) -> typing.Callable:
    """
    Create __getattr__ of uninstantiated module.

    Will return values from `uninstantiated` network if exist, if not
    will check it's `instantiated` network (if it exists), otherwise
    return `NoAttributeError`.

    Can be considered proxy passing user calls `module` after instantiation.

    Parameters
    ----------
    module : str
        Name of variable where instantiated module will be saved. Usually equal to
        global variable `MODULE`

    Returns
    -------
    Callable
        __getattr__ function
    """

    def __getattr__(self, name) -> str:
        if name == module:
            return super(type(self), self).__getattr__(name)
        return getattr(getattr(self, module), name)

    return __getattr__


# FIXED FOR PyTorch 1.4.0, 1.2.0 should work fine as well although it may throw warnings

# For warning regarding inability to find source code
# https://github.com/pytorch/pytorch/blob/master/torch/_utils_internal.py#L44

# getsourcefile
# https://github.com/python/cpython/blob/master/Lib/inspect.py#L692
# getfile
# https://github.com/python/cpython/blob/master/Lib/inspect.py#L654
# Simulate self.__module__.__file__ variable appropriately

# getsourcelines
# https://github.com/python/cpython/blob/master/Lib/inspect.py#L958


def create_reduce(module, parsed_init_arguments):
    """
    Create __reduce__ of uninstantiated module.

    This is one of core functionalities. Using custom `__reduce__` modules
    are reduced to their instantiated (after shape inference) versions.

    Due to this operation and `torchlayers.build` zero-overhead of shape
    inference is provided and support of `torchscript`.

    If the module was not instantiated `ValueError` error is thrown.

    Parameters
    ----------
    module : str
        Name of variable where instantiated module will be saved. Usually equal to
        global variable `MODULE`

    parsed_init_arguments : List[str]
        __init__ arguments parsed by `parse_arguments` function. Used to get names
        of variables saved in non-instantiated module to be passed to `__init__`
        of to be instantiated module.

    Returns
    -------
    Callable
        __getattr__ function
    """

    def __reduce__(self):
        infered_module = getattr(self, module, None)
        if infered_module is None:
            raise ValueError(
                "Model cannot be pickled as it wasn't instantiated. Pass example input through this module."
            )

        custom_reduce = getattr(infered_module, "__reduce__", None)
        if custom_reduce is None:
            raise ValueError(
                "Infered module does not have a __reduce__ method. Does it inherit from torch.nn.Module?"
            )
        return custom_reduce()

    return __reduce__
