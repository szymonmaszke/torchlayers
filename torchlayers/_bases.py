import typing

import torch


class InferDimension(torch.nn.Module):
    def __init__(
        self, *, module_name: str, instance_creator: typing.Callable = None, **kwargs
    ):
        self._module_name: str = module_name
        self._instance_creator = (
            instance_creator
            if instance_creator is not None
            else lambda _, inner_class, **kwargs: inner_class(**kwargs)
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._noninferable_parameters = (key for key in kwargs)
        super().__init__()

    def extra_repr(self):
        return ", ".join([f"{name}={value}" for name, value in self._kwargs.items()])

    def forward(self, inputs):
        if not hasattr(self, "_inner_module"):
            dimensions = len(inputs.shape)
            if dimensions < 2:
                inner_class = getattr(torch.nn, f"{self._module_name}1d", None)
            else:
                inner_class = getattr(
                    torch.nn, f"{self._module_name}{dimensions - 2}d", None
                )
            if inner_class is None:
                raise ValueError(
                    f"{self.module_name} could not be inferred from shape. "
                    "Only 5, 4, 3 and 2 (in case of normalization) "
                    f"dimensional input allowed, got {len(inputs.shape)}."
                )
            self.add_module(
                "_inner_module",
                self._instance_creator(
                    inputs,
                    inner_class,
                    **{
                        key: self.__dict__[key] for key in self._noninferable_parameters
                    },
                ),
            )

        return self._inner_module(inputs)
