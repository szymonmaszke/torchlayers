import collections

from .. import _inferable


def remove_right_side(argument):
    return argument.split("=")[0]


def remove_type_hint(argument):
    return argument.split(":")[0]


def get_per_module_index(module):
    if type(module).__name__ in _inferable.torch.recurrent:
        return 2
    return 1


def create_vars(self, **other_values):
    dictionary = collections.OrderedDict(vars(self))
    for key, value in other_values.items():
        key = remove_type_hint(key)
        dictionary.update({key: value})
        dictionary.move_to_end(key, last=False)

    return (
        (key, value)
        for key, value in dictionary.items()
        if not key.startswith("_") and key != "training"
    )


def process_arguments(*arguments):
    processed_arguments = [
        remove_type_hint(remove_right_side(argument)) for argument in arguments
    ]
    return processed_arguments[0], processed_arguments[1:]
