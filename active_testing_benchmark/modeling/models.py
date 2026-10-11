"""ResNet-18 model factory for a image-classification task."""

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import v2


def create_model(num_classes: int, pretrained: bool = True, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Create the supported ResNet-18 classifier."""

    model = resnet18(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    return model


def create_image_transform(image_size: int, training: bool) -> Callable:
    transforms: list[Callable] = []

    if training:
        transforms.append(v2.RandomHorizontalFlip())

    transforms.extend([
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return v2.Compose(transforms)
