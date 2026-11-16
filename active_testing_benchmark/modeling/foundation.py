"""Retinal foundation models used as candidate PPAT proxies, without fine-tuning.

PPAT uses a proxy that is stronger than the model under test and never fine-tuned on
the task: zero-shot CLIP for images and TabPFN with the training data as context
(Ashouritaklimi et al., 2026, Sec. 5). The retinal analogues here are

- FLAIR (Silva-Rodriguez et al., 2025), a vision-language model for fundus images,
  used zero-shot or as a frozen image encoder for a linear probe, and
- RETFound (Zhou et al., 2023), a masked-autoencoder ViT-L/16 pretrained on colour
  fundus photographs, used as a frozen encoder for a linear probe.

Both were pretrained on colour fundus photographs, while FairVision provides SLO
fundus images, so their transfer has to be measured (see
`experiments.proxy_experiment`). Neither package nor weights are project
dependencies. The classes import them lazily and explain what is missing.
"""

import argparse

from loguru import logger
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

FLAIR_REPOSITORY = "jusiro2/FLAIR"
FLAIR_IMAGE_SIZE = 512
RETFOUND_REPOSITORY = "YukunZhou/RETFound_mae_natureCFP"
RETFOUND_FILENAME = "RETFound_mae_natureCFP.pth"

# Zero-shot prompts per FairVision class index. FLAIR averages the text embeddings of
# the prompts of one class, as in its own `compute_text_embeddings`. FairVision's
# "vision threatening DR" covers severe non-proliferative and proliferative DR. These
# prompts are a research choice and should be reported with the results.
FLAIR_PROMPTS: dict[str, dict[int, list[str]]] = {
    "dr": {
        0: [
            "no diabetic retinopathy",
            "mild diabetic retinopathy",
            "moderate diabetic retinopathy",
        ],
        1: ["severe diabetic retinopathy", "proliferative diabetic retinopathy"],
    },
    "glaucoma": {0: ["normal", "no glaucoma"], 1: ["glaucoma"]},
    "amd": {
        0: ["normal", "no age-related macular degeneration"],
        1: ["early age-related macular degeneration"],
        2: ["intermediate age-related macular degeneration"],
        3: ["late age-related macular degeneration", "advanced age-related macular degeneration"],
    },
}


class ResizeLongestAndPad(nn.Module):
    """Resize the longest side to `size` and zero-pad the bottom and right to a square.

    This reproduces FLAIR's own `preprocess_image`, which keeps the aspect ratio and
    pads to `image_size` x `image_size`.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        scale = self.size / max(height, width)
        resized = TF.resize(image, [round(height * scale), round(width * scale)], antialias=True)
        return TF.pad(
            resized, [0, 0, self.size - resized.shape[-1], self.size - resized.shape[-2]]
        )


def flair_image_transform() -> v2.Compose:
    """FLAIR's input: RGB in [0, 1] at 512 x 512, with no mean or std normalisation."""
    return v2.Compose(
        [ResizeLongestAndPad(FLAIR_IMAGE_SIZE), v2.ToDtype(torch.float32, scale=True)]
    )


def _load_flair() -> nn.Module:
    try:
        from flair import FLAIRModel
    except ImportError as error:
        raise ImportError(
            "FLAIR is not installed. Add it from its repository, not the unrelated 'flair' "
            "package on PyPI, for example: uv run --with "
            "'flair @ git+https://github.com/jusiro/FLAIR.git' python -m ..."
        ) from error
    return FLAIRModel.from_pretrained(FLAIR_REPOSITORY)


class FlairImageEncoder(nn.Module):
    """FLAIR's projected image embedding, as frozen features for a linear probe.

    Expects `flair_image_transform` inputs. The model is kept in evaluation mode and
    never trained.
    """

    def __init__(self, flair: nn.Module | None = None) -> None:
        super().__init__()
        self.flair = flair if flair is not None else _load_flair()
        self.flair.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "FlairImageEncoder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.flair.vision_model(images)


class FlairZeroShot(nn.Module):
    """FLAIR zero-shot class log-probabilities from one prompt list per class.

    Mirrors `FLAIRModel.forward` for a batch: image and text embeddings, then FLAIR's
    scaled similarity logits, with the prompt embeddings of each class averaged first.
    """

    def __init__(self, prompts: dict[int, list[str]], flair: nn.Module | None = None) -> None:
        super().__init__()
        if sorted(prompts) != list(range(len(prompts))) or any(not p for p in prompts.values()):
            raise ValueError("Give at least one prompt for every class index 0..C-1")
        self.prompts = prompts
        self.encoder = FlairImageEncoder(flair)
        self._class_embeddings: torch.Tensor | None = None

    def class_embeddings(self, device: torch.device) -> torch.Tensor:
        if self._class_embeddings is None or self._class_embeddings.device != device:
            flair = self.encoder.flair
            embeddings = []
            with torch.no_grad():
                for class_index in range(len(self.prompts)):
                    input_ids, attention_mask = flair.preprocess_text(self.prompts[class_index])
                    text = flair.text_model(input_ids.to(device), attention_mask.to(device))
                    embeddings.append(text.mean(dim=0))
            self._class_embeddings = torch.stack(embeddings)
        return self._class_embeddings

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.encoder(images)
        logits = self.encoder.flair.compute_logits(
            image_embeddings, self.class_embeddings(image_embeddings.device)
        )
        return logits.log_softmax(dim=1)


class RetFoundFeatures(nn.Module):
    """Frozen RETFound (MAE, colour fundus) ViT-L/16 features, average-pooled.

    The encoder weights come from the gated Hugging Face repository
    `YukunZhou/RETFound_mae_natureCFP` (about 4 GB, including the MAE decoder, which is
    discarded). Request access on its page and log in with `hf auth login` first.
    The patch tokens are averaged after the pretrained final LayerNorm, so every loaded
    weight is pretrained. Inputs are 224 x 224 and ImageNet-normalised, which is the
    project's standard transform.
    """

    def __init__(self) -> None:
        super().__init__()
        from huggingface_hub import hf_hub_download
        import timm

        self.encoder = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            fc_norm=False,
        )
        path = hf_hub_download(RETFOUND_REPOSITORY, RETFOUND_FILENAME)
        # MAE checkpoints store their training arguments as an argparse Namespace.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = {
            key: value
            for key, value in checkpoint["model"].items()
            if not key.startswith(("decoder", "mask_token"))
        }
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"RETFound checkpoint is missing encoder weights: {missing}")
        if unexpected:
            logger.warning(f"Ignored RETFound checkpoint entries: {unexpected}")
        self.encoder.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "RetFoundFeatures":
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)
