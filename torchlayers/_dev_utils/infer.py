import inspect
import pickle

from ._infer_helpers import (create_vars, get_per_module_index,
                             process_arguments, remove_right_side,
                             remove_type_hint)

MODULE = "_inferred_module"
MODULE_CLASS = "_inferred_class_module"


def create_init(init_arguments):
    namespace = {}

    function = f"""def __init__(self, {", ".join(init_arguments)}):"""
    function += "\n super(type(self), self).__init__()\n "
    function += " ".join(
        [
            f"self.{remove_right_side(argument)} = {remove_type_hint(remove_right_side(argument))}\n"
            for argument in init_arguments
        ]
    )

    exec(function, namespace)
    return namespace["__init__"]


# Change forward to exec in dictionary so varargs are not used but exact same
# signature as original function
def create_forward(module, module_class, init_arguments):
    cleaned_arguments = [
        remove_type_hint(remove_right_side(argument)) for argument in init_arguments
    ]

    def forward(self, inputs, *args, **kwargs):
        inferred_module = getattr(self, module, None)
        if inferred_module is None:
            module_cls = getattr(self, module_class)
            self.add_module(
                module,
                module_cls(
                    inputs.shape[get_per_module_index(module_cls)],
                    *[getattr(self, argument) for argument in cleaned_arguments],
                ),
            )

            inferred_module = getattr(self, module)

        return inferred_module(inputs, *args, **kwargs)

    return forward


def create_repr(module, **non_inferable_names):
    def __repr__(self) -> str:
        inferred_module = getattr(self, module, None)
        if inferred_module is None:
            parameters = ", ".join(
                f"{key}={value}"
                for key, value in create_vars(self, non_inferable_names)
            )
            return f"{type(self).__name__}({parameters})"

        return repr(inferred_module)

    return __repr__


def create_getattr(module):
    def __getattr__(self, name) -> str:
        if name == module:
            return super(type(self), self).__getattr__(name)
        return getattr(getattr(self, module), name)

    return __getattr__


# For warning regarding inability to find source code
# https://github.com/pytorch/pytorch/blob/master/torch/_utils_internal.py#L44

# getsourcefile
# https://github.com/python/cpython/blob/master/Lib/inspect.py#L692
# getfile
# https://github.com/python/cpython/blob/master/Lib/inspect.py#L654
# Simulate self.__module__.__file__ variable appropriately

# getsourcelines
# https://github.com/python/cpython/blob/master/Lib/inspect.py#L958
def create_reduce(module, init_arguments):
    inferred_input, non_inferrable_arguments = process_arguments(init_arguments)

    def __reduce__(self):
        inferred_module = getattr(self, module, None)
        if inferred_module is None:
            raise ValueError(
                "Model cannot be pickled as it wasn't instantiated. Pass example input through this module."
            )

        custom_reduce = getattr(inferred_module, "__reduce__", None)
        if custom_reduce is None:
            return (
                type(inferred_module),
                (getattr(inferred_module, inferred_input),)
                + tuple(getattr(self, arg) for arg in non_inferrable_arguments),
                inferred_module.state_dict(),
            )
        return custom_reduce()

    return __reduce__
