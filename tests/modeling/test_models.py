import torch

from active_testing_benchmark.modeling import models


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
