import torch


def hard_sigmoid(tensor: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """
    Applies HardSigmoid function element-wise.

    See :class:`torchlayers.activations.HardSigmoid` for more details.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise
    inplace : bool, optional
        Whether operation should be performed `in-place`. Default: `False`

    Returns
    -------
    torch.Tensor
    """
    return torch.nn.functional.hardtanh(tensor, min_val=0, inplace=inplace)


class HardSigmoid(torch.nn.Module):
    """
    Applies HardSigmoid function element-wise.

    Uses `torch.nn.functional.hardtanh` internally with `0` and `1` ranges.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise

    """

    def forward(self, tensor: torch.Tensor):
        return hard_sigmoid(tensor)


def swish(tensor: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Applies Swish function element-wise.

    See :class:`torchlayers.activations.Swish` for more details.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise
    beta : float, optional
        Multiplier used for sigmoid. Default: 1.0 (no multiplier)

    Returns
    -------
    torch.Tensor
    """
    return torch.sigmoid(beta * tensor) * tensor


class Swish(torch.nn.Module):
    r"""
    Applies Swish function element-wise.

    .. math::

        Swish(x) = x / (1 + \exp(-beta * x))

    This form was originally proposed by Prajit Ramachandran et al. in
    `Searching for Activation Functions <https://arxiv.org/pdf/1710.05941.pdf>`__

    Parameters
    ----------
    beta : float, optional
        Multiplier used for sigmoid. Default: 1.0 (no multiplier)

    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, tensor: torch.Tensor):
        return swish(tensor, self.beta)


def hard_swish(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies HardSwish function element-wise.

    See :class:`torchlayers.activations.HardSwish` for more details.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise

    Returns
    -------
    torch.Tensor
    """
    return tensor * torch.nn.functional.relu6(tensor + 3) / 6


class HardSwish(torch.nn.Module):
    r"""
    Applies HardSwish function element-wise.

    .. math::

        HardSwish(x) = x * \min(\max(0,x + 3), 6) / 6


    While similar in effect to `Swish` should be more CPU-efficient.
    Above formula proposed by in Andrew Howard et al. in `Searching for MobileNetV3 <https://arxiv.org/pdf/1905.02244.pdf>`__.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor activated element-wise

    """

    def forward(self, tensor: torch.Tensor):
        return hard_swish(tensor)
