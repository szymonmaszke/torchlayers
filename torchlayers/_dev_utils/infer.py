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
        if not hasattr(self, module):
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

        return getattr(self, module)(inputs, *args, **kwargs)

    return forward


def create_repr(module, **non_inferable_names):
    def __repr__(self) -> str:
        if hasattr(self, module):
            return repr(getattr(self, module))

        parameters = ", ".join(
            f"{key}={value}" for key, value in create_vars(self, **non_inferable_names)
        )
        return f"{type(self).__name__}({parameters})"

    return __repr__
