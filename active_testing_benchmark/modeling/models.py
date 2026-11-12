"""ResNet-18 model factory for a image-classification task."""

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.transforms import v2


class FrozenImageNetFeatures(nn.Module):
    """Return 512 pooled ResNet-18 features without fitting any backbone weights.

    Expects RGB float BCHW inputs already resized and ImageNet-normalized by
    create_image_transform. We retain that shared image area instead of applying
    the weights' standard resize/center crop a second time. The weights are
    downloaded by torchvision on first use. Batch normalization always uses its
    pretrained running statistics, including when a parent forest enters train mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenImageNetFeatures":
        """Keep the frozen backbone in inference mode when a parent is trained."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def create_model(model_type="resnet", out_dim=1, use_pretrained=True):
    """Create model of specified type with or without pretrained weights"""

    if model_type == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    elif model_type == "efficientnet":
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        model.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=False)

    else:
        raise ValueError("Only 'resnet' and 'efficientnet'")

    return model


def create_image_transform(image_size: int, training: bool) -> Callable:
    transforms: list[Callable] = []

    if training:
        transforms.append(v2.RandomHorizontalFlip())

    transforms.extend(
        [
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return v2.Compose(transforms)
