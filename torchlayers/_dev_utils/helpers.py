import collections
import itertools
import typing


def is_kwarg(argument: str) -> bool:
    """`True` if argument name is `**kwargs`"""
    return "**" in argument


def is_vararg(argument: str) -> bool:
    """`True` if argument name is `*args`"""
    return "*" in argument and "**" not in argument


def remove_vararg(argument: str) -> str:
    """`True` if argument name is `*args`"""
    return argument.split("*")[1]


def remove_kwarg(argument: str) -> str:
    """Removes `**` in kwargs so those can be assigned in dynamic `__init__` creation"""
    return argument.split("**")[1]


def remove_right_side(argument: str) -> str:
    """Removes default arguments so their names can be used as values or names of variables."""
    return argument.split("=")[0]


def remove_type_hint(argument: str) -> str:
    """Removes any Python type hints.

    Those are incompatible with `exec` during dynamic `__init__` creation
    (at least not without major workarounds).

    Default values (right hand side) is preserved if exists.

    """
    splitted_on_type_hint = argument.split(":")
    no_type_hint = splitted_on_type_hint[0]
    if len(splitted_on_type_hint) > 1:
        splitted_on_default = splitted_on_type_hint[1].split("=")
        if len(splitted_on_default) > 1:
            no_type_hint += "={}".format(splitted_on_default[1])
    return no_type_hint


def create_vars(
    self,
    non_inferable_names: typing.Dict[str, typing.Any],
    varargs_variable: str,
    kwargs_variable: str,
) -> typing.List[str]:
    """
    Create list of arguments for uninstantiated `__repr__` module.

    Parameters
    ----------
    **non_inferable_names : Dict[str, Any]
        Non-inferable names and their respective values of the module
    varargs_variable : str
        Name of variable possibly holding varargs for module's __init__.
    kwargs_variable : str
        Name of variable possibly holding kwargs for module's __init__.

    Returns
    -------
    typing.List[str]:
        List of strings formatted in the manner argument=value to display
        for uninstantiated module.
    """
    dictionary = {**non_inferable_names, **collections.OrderedDict(vars(self))}

    args = [
        "{}={}".format(key, value)
        for key, value in dictionary.items()
        if not key.startswith("_") and key != "training"
    ]
    varargs = getattr(self, varargs_variable, None)
    if varargs is not None:
        args += [str(var) for var in varargs]

    kwargs = getattr(self, kwargs_variable, None)
    if kwargs is not None:
        args += ["{}={}".format(key, value) for key, value in kwargs.items()]

    return args
