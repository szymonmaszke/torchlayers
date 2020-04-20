import typing

import torch

from . import _dev_utils


class InferDimension(torch.nn.Module):
    """Infer dimensionality of module from input using dispatcher.

    Users can pass provide their own modules to infer dimensionality
    from input tensor by inheriting from this module and providing
    `super().__init__()` with `dispatcher` method::


        import torchlayers as tl

        class BatchNorm(tl.InferDimension):
            def __init__(
                self,
                num_features: int,
                eps: float = 1e-05,
                momentum: float = 0.1,
                affine: bool = True,
                track_running_stats: bool = True,
            ):
                super().__init__(
                    dispatcher={
                        # 5 dimensional tensor -> create torch.nn.BatchNorm3d
                        5: torch.nn.BatchNorm3d,
                        # 4 dimensional tensor -> create torch.nn.BatchNorm2d
                        4: torch.nn.BatchNorm2d,
                        3: torch.nn.BatchNorm1d,
                        2: torch.nn.BatchNorm1d,
                    },
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                )

    All dimension-agnostic modules in `torchlayers` are created this way.
    This class can also be mixed with `torchlayers.infer` for dimensionality
    and shape inference in one.

    This class works correctly with `torchlayers.build` and other provided
    functionalities.

    Parameters
    ----------
    dispatcher: Dict[int, torch.nn.Module]
        Key should be length of input's tensor shape. Value should be a `torch.nn.Module`
        to be used for the dimensionality.
    initializer: Callable[[torch.nn.Module, torch.Tensor, **kwargs], torch.nn.Module], optional
        How to initialize dispatched module. Can be used to modify it's creation.
        First argument - dispatched module class, second - input tensor, **kwargs are
        arguments to use for module initialization. Should return module's instance.
        By default dispatched module initialized with **kwargs is returned.
    **kwargs:
        Arguments used to initialize dispatched module

    """

    def __init__(
        self,
        dispatcher: typing.Dict[int, torch.nn.Module],
        initializer: typing.Callable = None,
        **kwargs,
    ):
        super().__init__()

        self._dispatcher = dispatcher
        self._inner_module_name = "_inner_module"

        if initializer is None:
            self._initializer = lambda dispatched_class, _, **kwargs: dispatched_class(
                **kwargs
            )
        else:
            self._initializer = initializer

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._noninferable_attributes = [key for key in kwargs]
        self._repr = _dev_utils.infer.create_repr(self._inner_module_name, **kwargs)
        self._reduce = _dev_utils.infer.create_reduce(
            self._inner_module_name, self._noninferable_attributes
        )

    def __repr__(self):
        return self._repr(self)

    def __reduce__(self):
        return self._reduce(self)

    def forward(self, inputs):
        module = getattr(self, self._inner_module_name, None)
        if module is None:
            dimensionality = len(inputs.shape)
            dispatched_class = self._dispatcher.get(dimensionality)
            if dispatched_class is None:
                dispatched_class = self._dispatcher.get("*")
                if dispatched_class is None:
                    raise ValueError(
                        "{} could not be inferred from shape. Got tensor of dimensionality: {} but only {} are allowed".format(
                            self._module_name,
                            dimensionality,
                            list(self._dispatcher.keys()),
                        )
                    )

            self.add_module(
                self._inner_module_name,
                self._initializer(
                    dispatched_class,
                    inputs,
                    **{
                        key: getattr(self, key) for key in self._noninferable_attributes
                    },
                ),
            )

        return getattr(self, self._inner_module_name)(inputs)
