"""Model factory for the image-classification benchmark."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create the supported ResNet-18 classifier."""
    model, _ = create_model_and_transform(name, num_classes, pretrained=pretrained, image_size=224)
    return model


def create_model_and_transform(
    name: str, num_classes: int, pretrained: bool, image_size: int
) -> tuple[nn.Module, Callable]:
    """Create ResNet-18 and its image preprocessing transform."""
    if name != "resnet18":
        raise ValueError(f"Unknown model {name!r}; choose resnet18")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, torchvision_transform(image_size, training=True)


def torchvision_transform(image_size: int, training: bool) -> Callable:
    transforms: list[Callable] = [v2.Resize((image_size, image_size))]
    if training:
        transforms.append(v2.RandomHorizontalFlip())
    transforms.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return v2.Compose(transforms)
