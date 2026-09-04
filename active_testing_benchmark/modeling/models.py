"""Model factory for the small image-classification benchmark."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
    densenet121,
    resnet18,
    vit_b_16,
)
from torchvision.transforms import v2


class SmallCNN(nn.Module):
    """Deliberately small baseline CNN."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images).flatten(1))


class CLIPClassifier(nn.Module):
    """Frozen OpenCLIP image encoder with a trainable linear classifier."""

    def __init__(self, image_encoder: nn.Module, embedding_size: int, num_classes: int) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            embeddings = self.image_encoder.encode_image(images)
        return self.classifier(embeddings.float())


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a supported classifier. CLIP requires ``open-clip-torch``."""
    model, _ = create_model_and_transform(name, num_classes, pretrained=pretrained, image_size=224)
    return model


def create_model_and_transform(
    name: str, num_classes: int, pretrained: bool, image_size: int
) -> tuple[nn.Module, Callable]:
    """Create a model and its image preprocessing transform."""
    if name == "cnn":
        return SmallCNN(num_classes), torchvision_transform(image_size, training=True)
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, torchvision_transform(image_size, training=True)
    if name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model, torchvision_transform(image_size, training=True)
    if name == "vit":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model, torchvision_transform(image_size, training=True)
    if name == "clip":
        return create_clip_model(num_classes, pretrained, image_size)
    supported = "cnn, resnet18, densenet121, vit, clip"
    raise ValueError(f"Unknown model {name!r}; choose one of: {supported}")


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


def create_clip_model(
    num_classes: int, pretrained: bool, image_size: int
) -> tuple[nn.Module, Callable]:
    """Create ViT-B/32 CLIP and use OpenCLIP's own preprocessing."""
    try:
        import open_clip
    except (
        ImportError
    ) as error:  # pragma: no cover - dependency is declared, but optional at import time
        raise ImportError("CLIP requires open-clip-torch. Run `uv sync` to install it.") from error

    checkpoint = "laion2b_s34b_b79k" if pretrained else None
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=checkpoint, force_image_size=image_size
    )
    return CLIPClassifier(clip_model, clip_model.visual.output_dim, num_classes), preprocess
