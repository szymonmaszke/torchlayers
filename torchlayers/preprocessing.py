import abc
import numbers
import random

import torch


class _GetInputs(torch.nn.Module):
    def _get_inputs(self, inputs):
        if self.inplace:
            return inputs
        return inputs.clone()


class RandomApply(_GetInputs):
    """Apply randomly a list of transformations with a given probability.


    .. note::
            **IMPORTANT**: This function is a `no-op` during inference phase
            (module in `eval` mode).

    Parameters
    ----------
    transforms : List | Tuple
        List of transformations
    p : float, optional
        Probability to apply list of transformations. Default: `0.5`
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`

    """

    def __init__(self, transforms, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, inputs):
        if not self.training:
            return inputs

        x = self._get_inputs(inputs)
        if random.random() > self.p:
            return inputs
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomChoice(torch.nn.Module):
    """Apply single transformation randomly picked from a list.

    .. note::
            **IMPORTANT**: This function is a `no-op` during inference phase
            (module in `eval` mode).

    Parameters
    ----------
    transforms : List | Tuple
        List of transformations
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, inputs):
        if not self.training:
            return inputs

        transform = random.choice(self.transforms)
        return transform(inputs)


class RandomOrder(_GetInputs):
    """Apply single transformation randomly picked from a list.

    .. note::
            **IMPORTANT**: This function is a `no-op` during inference phase
            (module in `eval` mode).

    Parameters
    ----------
    transforms : List | Tuple
        List of transformations
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`
    """

    def __init__(self, transforms, inplace: bool = False):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
        self.inplace: bool = inplace

    def forward(self, inputs):
        if not self.training:
            return inputs

        x = self._get_inputs(inputs)

        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            x = self.transforms[i](x)
        return x


class Normalize(torch.nn.Module):
    """Normalize batch of tensor images with mean and standard deviation.

    Given mean values: `(M1,...,Mn)` and std values: `(S1,..,Sn)` for `n` channels
    (or other broadcastable to `n` values),
    this transform will normalize each channel of tensors in batch via formula:
    `output[channel] = (input[channel] - mean[channel]) / std[channel]`

    Parameters
    ----------
    mean : Tuple | List | torch.tensor
        Sequence of means for each channel
    std : Tuple | List | torch.tensor
        Sequence of means for each channel
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`

    """

    @classmethod
    def _transform_to_tensor(cls, tensor, name: str):
        if not torch.is_tensor(tensor):
            if isinstance(tensor, (tuple, list)):
                return torch.tensor(tensor)
            else:
                raise ValueError(
                    "{} is not an instance of either list, tuple or torch.tensor.".format(
                        name
                    )
                )
        return tensor

    @classmethod
    def _check_shape(cls, tensor, name):
        if len(tensor.shape) > 1:
            raise ValueError(
                "{} should be 0 or 1 dimensional tensor. Got {} dimensional tensor.".format(
                    name, len(tensor.shape)
                )
            )

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, inplace: bool = False):
        tensor_mean = Normalize._transform_to_tensor(mean, "mean")
        tensor_std = Normalize._transform_to_tensor(std, "std")
        Normalize._check_shape(tensor_mean, "mean")
        Normalize._check_shape(tensor_std, "std")

        if torch.any(tensor_std == 0):
            raise ValueError(
                "One or more std values are zero which would lead to division by zero."
            )

        super().__init__()

        self.register_buffer("mean", tensor_mean)
        self.register_buffer("std", tensor_std)
        self.inplace: bool = inplace

    def forward(self, inputs):
        inputs_length = len(inputs.shape) - 2
        mean = self.mean.view(1, -1, *([1] * inputs_length))
        std = self.std.view(1, -1, *([1] * inputs_length))
        if self.inplace:
            inputs.sub_(mean).div_(std)
            return inputs
        return (inputs - mean) / std


class Transform(_GetInputs):
    """{header}

    {body}

    .. note::
            **IMPORTANT**: This function is a `no-op` during inference phase
            (module in `eval` mode).

    Parameters
    ----------
    p : float, optional
        Probability of applying transformation. Default: `0.5`
    batch : bool, optional
        Whether this operation should be applied on whole batch.
        If `True` the same transformation is applied on whole batch (or to
        no image in batch at all). If `False` apply transformation to `p` percent
        of images contained in batch at random. Default: `False`
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`

    """

    def __init__(self, p: float = 0.5, batch: bool = False, inplace: bool = False):
        if p < 0 or p > 1:
            raise ValueError("Probability of rotation should be between 0 and 1")
        super().__init__()
        self.p = p
        self.batch: bool = batch
        self.inplace: bool = inplace

    def forward(self, inputs):
        if self.training:
            x = self._get_inputs(inputs)
            if self.batch:
                if random.random() < self.p:
                    return self.transform(x)
            else:
                indices = torch.randperm(x.shape[0])[: int(x.shape[0] * self.p)]
                x[indices] = self.transform(x[indices])
                return x

        return inputs

    @abc.abstractmethod
    def transform(self, x):
        pass


class _RandomRotate90(Transform):
    """Randomly rotate image {} by `90` degrees `k` times.

    Rotation will be done in a clockwise manner.

    .. note::
            **IMPORTANT**: This function is a `no-op` during inference phase.

    Parameters
    ----------
    p : float, optional
        Probability of applying transformation. Default: `0.5`
    k : int, optional
        Number of times to rotate. Default: `1`
    batch : bool, optional
        Whether this operation should be applied on whole batch.
        If `True` the same transformation is applied on whole batch (or to
        no image in batch at all). If `False` apply transformation to `p` percent
        of images contained in batch at random. Default: `False`
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`

    """

    def __init__(
        self, p: float = 0.5, k: int = 1, batch: bool = False, inplace: bool = False
    ):
        super().__init__(p, batch, inplace)
        self.k: int = k


class ClockwiseRandomRotate90(_RandomRotate90):
    __doc__ = _RandomRotate90.__doc__.format("clockwise")

    def transform(self, x):
        return torch.rot90(x, k=self.k, dims=(-1, -2))


class AntiClockwiseRandomRotate90(_RandomRotate90):
    __doc__ = _RandomRotate90.__doc__.format("anticlockwise")

    def transform(self, x):
        return torch.rot90(x, k=self.k, dims=(-2, -1))


class RandomHorizontalFlip(Transform):
    __doc__ = Transform.__doc__.format(
        header="Randomly perform horizontal flip on batch of images.", body=""
    )

    def transform(self, x):
        return torch.flip(x, dims=(-1,))


class RandomVerticalFlip(Transform):
    __doc__ = Transform.__doc__.format(
        header="Randomly perform vertical flip on batch of images.", body=""
    )

    def transform(self, x):
        return torch.flip(x, dims=(-2,))


class RandomVerticalHorizontalFlip(Transform):
    __doc__ = Transform.__doc__.format(
        header="Randomly perform vertical and horizontal flip on batch of images.",
        body="",
    )

    def transform(self, x):
        return torch.flip(x, dims=(-2, -1))


class RandomErasing(Transform):
    """Randomly select rectangle regions in a batch of image and erase their pixels.

    Originally proposed by Zhong et al. in `Random Erasing Data Augmentation <https://arxiv.org/pdf/1708.04896.pdf>`__

    .. note::
            **IMPORTANT**: This function is a `no-op` during inference phase.
    .. note::
            Each image in batch will have the same rectangle region cut out
            due to efficiency reasons. It probably doesn't alter the idea
            drastically but exact effects weren't tested.


    Parameters
    ----------
    max_rectangles : int
        Maximum number of rectangles to create.
    max_height : int
        Maximum height of the rectangle.
    max_width : int, optional
        Maximum width of the rectangle. Default: same as `max_height`
    min_rectangles : int, optional
        Minimum number of rectangles to create. Default: same as `max_rectangles`
    min_height : int, optional
        Minimum height of the rectangle. Default: same as `max_height`
    min_width : int, optional
        Minimum width of the rectangle. Default: same as `min_width`
    fill : Callable, optional
        Callable used to fill the rectangle. It will be passed three arguments:
        `size` (as a `tuple`), `dtype`, `layout` and `device` of original tensor.
        If you want to specify `random uniform` filling you can use this:
        `lambda` function: `random = lambda size, dtype, layout, device: torch.randn(*size, dtype=dtype, layout=layout, device=device)`.
        If non-default is used, users are responsible for ensuring correct tensor format based
        on callable passed arguments.
        Default: fill with `0.0`.
    p : float, optional
        Probability of applying transformation. Default: `0.5`
    batch : bool, optional
        Whether this operation should be applied on whole batch.
        If `True` the same transformation is applied on whole batch (or to
        no image in batch at all). If `False` apply transformation to `p` percent
        of images contained in batch at random. Default: `False`
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`

    """

    @classmethod
    def _check_min_max(cls, minimum, maximum, name: str):
        if minimum > maximum:
            raise ValueError(
                "{} minimum is greater than maximum. Got minimum: {} and maximum: {}.".format(
                    name.capitalize(), minimum, maximum
                )
            )

    @classmethod
    def _check_greater_than_zero(cls, value, name: str):
        if value <= 0:
            raise ValueError(
                "Minimal {} should be greater than 0. Got {}".format(name, value)
            )

    @classmethod
    def _conditional_default(cls, value, default):
        if value is None:
            return default
        return value

    def __init__(
        self,
        max_rectangles: int,
        max_height: int,
        max_width: int = None,
        min_rectangles: int = None,
        min_height: int = None,
        min_width: int = None,
        fill=None,
        p: float = 0.5,
        batch: bool = False,
        inplace: bool = False,
    ):
        RandomErasing._check_greater_than_zero(min_rectangles, "holes")
        RandomErasing._check_greater_than_zero(min_height, "height")
        RandomErasing._check_greater_than_zero(min_width, "width")

        RandomErasing._check_min_max(min_rectangles, max_rectangles, "holes")
        RandomErasing._check_min_max(min_width, max_width, "width")
        RandomErasing._check_min_max(min_height, max_height, "height")

        self.max_rectangles: int = max_rectangles
        self.max_height: int = max_height

        self.max_width = RandomErasing._conditional_default(max_width, max_height)
        self.min_rectangles = RandomErasing._conditional_default(
            min_rectangles, max_rectangles
        )
        self.min_height = RandomErasing._conditional_default(min_height, max_height)
        self.min_width = RandomErasing._conditional_default(min_width, self.max_width)

        self.fill = RandomErasing._conditional_default(fill, lambda *_: 0.0)

        super().__init__(p, batch, inplace)

    def transform(self, x):
        holes = random.randint(self.min_rectangles, self.max_rectangles)

        start_hs = torch.randint(0, x.shape[-2] - self.max_height, (holes,))
        start_ws = torch.randint(0, x.shape[-1] - self.max_width, (holes,))
        heights = torch.randint(self.min_height, self.max_height, (holes,))
        widths = torch.randint(self.min_width, self.max_width, (holes,))

        for start_h, start_w, height, width in zip(start_hs, start_ws, heights, widths):
            x[..., start_h : start_h + height, start_w : start_w + width] = self.fill(
                (*(x.shape[:-2]), start_h + height, start_w + width),
                x.dtype,
                x.layout,
                x.device,
            )

        return x
