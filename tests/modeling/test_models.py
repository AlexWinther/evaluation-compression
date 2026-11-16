import argparse
import sys

import pytest
import torch
from torch import nn

from active_testing_benchmark.modeling import foundation, models


def test_imagenet_features_are_frozen_and_keep_batch_norm_statistics(monkeypatch):
    requested_weights = []
    original = models.resnet18

    def offline_backbone(*, weights):
        requested_weights.append(weights)
        return original(weights=None)

    monkeypatch.setattr(models, "resnet18", offline_backbone)
    extractor = models.FrozenImageNetFeatures()
    extractor.train()
    before = {name: value.clone() for name, value in extractor.named_buffers()}
    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    features = extractor(images)
    assert requested_weights == [models.ResNet18_Weights.IMAGENET1K_V1]
    assert features.shape == (2, 512)
    assert not features.requires_grad
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    assert all(not module.training for module in extractor.modules())
    for name, value in extractor.named_buffers():
        torch.testing.assert_close(value, before[name])
    torch.testing.assert_close(extractor(images), features)


class FakeFlair(nn.Module):
    """A stand-in for FLAIRModel with its batch API: image, text and logit methods."""

    def __init__(self):
        super().__init__()
        self.vision_model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.prompts = []

    def preprocess_text(self, prompts):
        self.prompts.extend(prompts)
        ids = torch.tensor([[len(prompt) % 3] for prompt in prompts])
        return ids, torch.ones_like(ids)

    def text_model(self, input_ids, attention_mask):
        return nn.functional.one_hot(input_ids[:, 0], 3).float()

    def compute_logits(self, image_embeddings, text_embeddings):
        return 10 * image_embeddings @ text_embeddings.t()


def test_flair_transform_keeps_aspect_ratio_and_pads_to_a_square():
    image = torch.full((3, 100, 50), 255, dtype=torch.uint8)
    transformed = foundation.flair_image_transform()(image)
    assert transformed.shape == (3, 512, 512)
    assert transformed.dtype == torch.float32
    assert transformed[:, :, :256].min() == 1.0
    assert transformed[:, :, 256:].max() == 0.0


def test_flair_zero_shot_averages_prompt_embeddings_per_class():
    flair = FakeFlair()
    # Prompt lengths 3 and 6 map to embedding 0, length 4 to embedding 1.
    zero_shot = foundation.FlairZeroShot({0: ["abc", "abcdef"], 1: ["abcd"]}, flair)
    images = torch.zeros(2, 3, 4, 4)
    images[0, 0] = 1.0
    images[1, 1] = 1.0
    log_probabilities = zero_shot(images)
    assert flair.prompts == ["abc", "abcdef", "abcd"]
    torch.testing.assert_close(log_probabilities.exp().sum(dim=1), torch.ones(2))
    assert log_probabilities.argmax(dim=1).tolist() == [0, 1]
    assert all(not parameter.requires_grad for parameter in flair.parameters())
    with pytest.raises(ValueError, match="every class"):
        foundation.FlairZeroShot({0: ["normal"], 2: ["glaucoma"]}, flair)


def test_missing_flair_package_explains_the_installation(monkeypatch):
    monkeypatch.setitem(sys.modules, "flair", None)
    with pytest.raises(ImportError, match="jusiro/FLAIR"):
        foundation.FlairImageEncoder()


def test_retfound_loads_encoder_weights_and_drops_the_decoder(monkeypatch, tmp_path):
    import huggingface_hub
    import timm

    create = timm.create_model

    def tiny_vit(name, **kwargs):
        assert name == "vit_large_patch16_224"
        assert kwargs["global_pool"] == "avg" and kwargs["fc_norm"] is False
        return create("vit_tiny_patch16_224", img_size=32, **kwargs)

    encoder = tiny_vit("vit_large_patch16_224", num_classes=0, global_pool="avg", fc_norm=False)
    state = {**encoder.state_dict(), "decoder_embed.weight": torch.zeros(1), "mask_token": 0}
    path = tmp_path / foundation.RETFOUND_FILENAME
    torch.save({"model": state, "args": argparse.Namespace(lr=1e-3)}, path)
    monkeypatch.setattr(timm, "create_model", tiny_vit)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, filename: str(path))
    features = foundation.RetFoundFeatures()
    images = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(features(images), encoder.eval()(images))
    assert all(not parameter.requires_grad for parameter in features.parameters())

    del state["norm.weight"]
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="missing encoder weights"):
        foundation.RetFoundFeatures()
