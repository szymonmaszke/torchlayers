import abc

import torch

from . import module


class StochasticDepth(torch.nn.Module):
    """Randomly skip module during training with specified `p`, leaving inference untouched.

    Originally proposed by Gao Huang et al. in
    `Deep Networks with Stochastic Depth <www.arxiv.org/abs/1512.03385>`__.

    Originally devised as regularization, though `other research <https://web.stanford.edu/class/cs331b/2016/projects/kaplan_smith_jiang.pdf>`__  suggests:

    - "[...] StochasticDepth Nets are less tuned for low-level feature extraction but more tuned for higher level feature differentiation."
    - "[...] Stochasticity does not help with the ”dead neurons” problem; in fact the problem is actually more pronounced in the early layers. Nonetheless, the Stochastic Depth Network has relatively fewer dead neurons in later layers."

    It might be useful to employ this technique to layers closer to the bottleneck.

    Example::


        import torchlayers as tl

        # Assume only 128 channels can be an input in this case
        block = tl.StochasticDepth(tl.Conv(128), p=0.3)
        # May skip tl.Conv with 0.3 probability
        block(torch.randn(1, 3, 32, 32))

    Parameters
    ----------
    module : torch.nn.Module
        Any module whose output might be skipped
        (output shape of it has to be equal to the shape of inputs).
    p : float, optional
        Probability of survival (e.g. the layer will be kept). Default: ``0.5``

    """

    def __init__(self, module: torch.nn.Module, p: float = 0.5):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError(
                "Stochastic Depth p has to be between 0 and 1 but got {}".format(p)
            )
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_():
            return inputs
        return self.p * self.module(inputs)


class Dropout(module.InferDimension):
    """Randomly zero out some of the tensor elements.

    .. note::
            Changes input only if `module` is in `train` mode.

    Based on input shape it either creates `2D` or `3D` version of dropout for inputs of shape
    `4D`, `5D` respectively (including batch as first dimension).
    For every other dimension, standard `torch.nn.Dropout` will be used.

    Parameters
    ----------
    p : float, optional
        Probability of an element to be zeroed. Default: ``0.5``
    inplace : bool, optional
        If ``True``, will do this operation in-place. Default: ``False``

    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__(
            dispatcher={
                5: torch.nn.Dropout3d,
                4: torch.nn.Dropout2d,
                "*": torch.nn.Dropout,
            },
            p=p,
            inplace=inplace,
        )


class StandardNormalNoise(torch.nn.Module):
    """Add noise from standard normal distribution during forward pass.

    .. note::
            Changes input only if `module` is in `train` mode.

    Example::


        import torchlayers as tl

        model = tl.Sequential(
            tl.StandardNormalNoise(), tl.Linear(10), tl.ReLU(), tl.tl.Linear(1)
        )
        tl.build(model, torch.randn(3, 30))

        # Noise from Standard Normal distribution will be added
        model(torch.randn(3, 30))

        model.eval()
        # Eval mode, no noise added
        model(torch.randn(3, 30))

    """

    def forward(self, inputs):
        if self.training:
            return inputs + torch.randn_like(inputs)
        return inputs


class UniformNoise(torch.nn.Module):
    """Add noise from uniform `[0, 1)` distribution during forward pass.

    .. note::
            Changes input only if `module` is in `train` mode.

    Example::


        import torchlayers as tl

        noisy_linear_regression = tl.Sequential(
            tl.UniformNoise(), tl.Linear(1)
        )
        tl.build(model, torch.randn(1, 10))

        # Noise from Uniform distribution will be added
        model(torch.randn(64, 10))

        model.eval()
        # Eval mode, no noise added
        model(torch.randn(64, 10))

    """

    def forward(self, inputs):
        if self.training:
            return inputs + torch.rand_like(inputs)
        return inputs


class WeightDecay(torch.nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay <= 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L2(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * parameter.data


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)
