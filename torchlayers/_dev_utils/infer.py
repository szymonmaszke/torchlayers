from ._infer_helpers import (create_vars, get_per_module_index,
                             remove_right_side, remove_type_hint)

MODULE = "_inferred_module"
MODULE_CLASS = "_inferred_class_module"


def create_init(*arguments):
    namespace = {}

    header = f"""def __init__(self, {", ".join(arguments)}):"""
    header += "\n super(type(self), self).__init__()\n "
    header += " ".join(
        [
            f"self.{remove_right_side(argument)} = {remove_type_hint(remove_right_side(argument))}\n"
            for argument in arguments
        ]
    )

    exec(header, namespace)
    return namespace["__init__"]


def create_forward(module, module_class, *arguments):
    def forward(self, inputs, *args, **kwargs):
        inferred_module = getattr(self, module, None)
        if inferred_module is None:
            module_cls = getattr(self, module_class)
            self.add_module(
                module,
                module_cls(
                    inputs.shape[get_per_module_index(module_cls)],
                    *[
                        getattr(self, remove_type_hint(remove_right_side(argument)))
                        for argument in arguments
                    ],
                ),
            )

            for argument in arguments:
                delattr(self, remove_type_hint(remove_right_side(argument)))

            inferred_module = getattr(self, module)

        return inferred_module(inputs, *args, **kwargs)

    return forward


def create_repr(module, **non_inferable_names):
    def __repr__(self) -> str:
        inferred_module = getattr(self, module, None)
        if inferred_module is None:
            parameters = ", ".join(
                f"{key}={value}"
                for key, value in create_vars(self, **non_inferable_names)
            )
            return f"{type(self).__name__}({parameters})"

        return repr(inferred_module)

    return __repr__


def create_getattr(module):
    def __getattr__(self, name) -> str:
        if name == module:
            # PyTorch's __getattr__ checking buffers etc.
            return super(type(self), self).__getattr__(name)
        # Inferred module getattr
        return getattr(getattr(self, module), name)

    return __getattr__
