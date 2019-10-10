import torch

__all__ = ["StochasticDepth"]


class StochasticDepth(torch.nn.Module):
    """Randomly skip module during training with specified `p`, leaving inference untouched.

    Originally proposed by Gao Huang et. al in
    `Deep Networks with Stochastic Depth <www.arxiv.org/abs/1512.03385>`__.

    Originally devised as regularization, though `other research <https://web.stanford.edu/class/cs331b/2016/projects/kaplan_smith_jiang.pdf>`__  suggests:

    - "[...] StochasticDepth Nets are less tuned for low-level feature extraction
    but more tuned for higher level feature differentiation."
    - "[...] Stochasticity does not help with the ”dead neurons” problem;
    in fact the problem is actually more pronounced in the early layers.
    Nonetheless, the Stochastic Depth Network has relatively fewer dead neurons in later layers."

    It might be useful to employ this technique onto layers closer to the bottleneck.

    Parameters
    ----------
    module: torch.nn.Module
            Any module whose output might be skipped
            (output shape of it has to be equal to the shape of inputs).
    p: float
            Probability to skip the module.

    """

    def __init__(self, module: torch.nn.Module, p: float):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError(
                "Stochastic Depth p has to be between 0 and 1, " f"but got {p}"
            )
        self.module: torch.nn.Module = module
        self.p: float = p
        self._sampler = torch.nn.Parameter(torch.Tensor(1), requires_grad=False)

    def forward(self, inputs):
        if self._sampler.uniform_() < self.p and self.training:
            return inputs
        return self.module(inputs)
