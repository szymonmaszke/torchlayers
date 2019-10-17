import typing

import torch

from . import infer


# Fix getattr and deletion of attributes
class InferDimension(torch.nn.Module):
    def __init__(self, *, instance_creator: typing.Callable = None, **kwargs):
        self._inner_module_name = "_inner_module"

        self._module_name: str = type(self).__name__
        self._instance_creator = (
            instance_creator
            if instance_creator is not None
            else lambda _, inner_class, **kwargs: inner_class(**kwargs)
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._noninferable_attributes = [key for key in kwargs]
        self._repr = infer.create_repr(self._inner_module_name, **kwargs)
        self._reduce = infer.create_reduce(
            self._inner_module_name, *self._noninferable_attributes
        )
        super().__init__()

    def __repr__(self):
        return self._repr(self)

    def __reduce__(self):
        return self._reduce(self)

    def _module_not_found(self, inputs):
        raise ValueError(
            f"{self._module_name} could not be inferred from shape. "
            f"Only 5, 4 or 3 dimensional input allowed (including batch dimension), got {len(inputs.shape)}."
        )

    def forward(self, inputs):
        module = getattr(self, self._inner_module_name, None)
        if module is None:
            dimensions = len(inputs.shape)
            inner_class = getattr(
                torch.nn, f"{self._module_name}{dimensions - 2}d", None
            )
            if inner_class is None:
                inner_class = self._module_not_found(inputs)

            self.add_module(
                self._inner_module_name,
                self._instance_creator(
                    inputs,
                    inner_class,
                    **{
                        key: getattr(self, key) for key in self._noninferable_attributes
                    },
                ),
            )

        return getattr(self, self._inner_module_name)(inputs)


class Representation(torch.nn.Module):
    def __repr__(self):
        parameters = ", ".join(
            [
                f"{key}={value}"
                for key, value in vars(self).items()
                if not key.startswith("_") and key != "training"
            ]
        )
        return f"{type(self).__name__}({parameters})"
