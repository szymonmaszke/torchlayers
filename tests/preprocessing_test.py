import torch
import torchvision

import pytest
import torchfunc
import torchlayers as tl

# torchvision.transforms.RandomHorizontalFlip(p=1.0) vs tl.RandomHorizontalFlip
# torchvision.transforms.RandomVerticalFlip(p=1.0) vs tl.RandomVerticalFlip
# torchvision.transforms.Normalize vs tl.Normalize with same mean and std
# torchvision.transforms.functional.rotate via Lambdas with all Rotate90's
# RandomHorizontalVerticalFlip with both from torchvision one after another

# RandomApply withh the same probability and same outputs
# RandomChoice via same seed?
# RandomOrder via same seed?


class TorchvisionRotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return torchvision.transforms.functional.to_tensor(
            torchvision.transforms.functional.rotate(x, self.angle)
        )


def generate_inputs():
    operations = [
        # Flipping
        (
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            tl.RandomHorizontalFlip(p=1.0),
        ),
        (
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomVerticalFlip(p=1.0),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            tl.RandomVerticalFlip(p=1.0),
        ),
        (
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomVerticalFlip(p=1.0),
                    torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            tl.RandomVerticalHorizontalFlip(p=1.0),
        ),
        # Normalization
        (
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            ),
            tl.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ),
        # Random 90 rotations
        (TorchvisionRotate(90), tl.AntiClockwiseRandomRotate90(p=1.0),),
        (TorchvisionRotate(-90), tl.ClockwiseRandomRotate90(p=1.0),),
        (TorchvisionRotate(180), tl.ClockwiseRandomRotate90(p=1.0, k=2),),
    ]

    dataset = lambda transform: torchvision.datasets.CIFAR10(
        root=".", download=True, transform=transform
    )
    for torchvision_transform, torchlayers_module in operations:
        yield torch.utils.data.DataLoader(
            dataset(torchvision_transform), batch_size=2, shuffle=False
        ), torch.utils.data.DataLoader(
            dataset(torchvision.transforms.ToTensor()), batch_size=2, shuffle=False
        ), torchlayers_module


@pytest.mark.parametrize(
    "torchvision_dataloader,dataloader,module", list(generate_inputs())
)
def test_functionality(torchvision_dataloader, dataloader, module):
    for i, ((torchvision_images, _), (images, _)) in enumerate(
        zip(torchvision_dataloader, dataloader)
    ):
        if i == 10:
            break
        transformed_images = module(images)
        assert torch.allclose(transformed_images, torchvision_images)


# @pytest.mark.parametrize(
#     "torchvision_dataloader,dataloader,module",
#     list(generate_inputs(different_batches=True)),
# )
# def test_time(torchvision_dataloader, dataloader, module):
#     THRESHOLD = 1
#     LIMIT = 50

#     @torchfunc.Timer()
#     def time_torchvision(limit):
#         total = 0
#         for i, (torchvision_images, _) in enumerate(torchvision_dataloader):
#             if i == limit:
#                 break
#             total += torchvision_images.mean()
#         return total

#     @torchfunc.Timer()
#     def time_torchlayers(limit):
#         total = 0
#         for i, (images, _) in enumerate(dataloader):
#             if i == limit:
#                 break
#             total += module(images).mean()
#         return total

#     torchvision_value, torchvision_time = time_torchvision(LIMIT)
#     torchlayers_value, torchlayers_time = time_torchlayers(LIMIT)

#     assert torch.allclose(torchvision_value, torchlayers_value)
#     assert THRESHOLD * torchlayers_time < torchvision_time
