from .. import _inferable


def _remove_default_value(argument):
    return argument.split("=")[0]


def _get_per_module_index(module):
    if type(module).__name__ in _inferable.torch.recurrent:
        return 2
    return 1


def _get_dictionary(self, input_name):
    if hasattr(self, "_inferred_module"):
        dictionary = vars(self._inferred_module)
    else:
        dictionary = collections.OrderedDict(vars(self))
        dictionary.update({input_name: "?"})
        dictionary.move_to_end(input_name, last=False)

    return (
        (key, value)
        for key, value in dictionary.items()
        if not key.startswith("_") and key != "training"
    )


def create_init(arguments):
    namespace = {}

    header = f"""def __init__(self, {", ".join(arguments)}):"""
    header += "\n super(type(self), self).__init__()\n "
    header += " ".join(
        [
            f"self.{_remove_default_value(argument)} = {_remove_default_value(argument)}\n"
            for argument in arguments
        ]
    )

    exec(header, namespace)
    return namespace["__init__"]


def create_forward(arguments):
    def forward(self, inputs, *args, **kwargs):
        if not hasattr(self, "_inferred_module"):
            self.add_module(
                "_inferred_module",
                self._inferred_module_class(
                    inputs.shape[_get_per_module_index(self._inferred_module_class)],
                    *[
                        getattr(self, _remove_default_value(argument))
                        for argument in arguments
                    ],
                ),
            )

            for argument in arguments:
                delattr(self, _remove_default_value(argument))

        return self._inferred_module(inputs, *args, **kwargs)

    return forward


def create_repr(input_name):
    def __repr__(self) -> str:
        parameters = ", ".join(
            f"{key}={value}" for key, value in _get_dictionary(self, input_name)
        )
        return f"{type(self).__name__}({parameters})"

    return __repr__
