import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from active_testing_benchmark.active_testing import (
    surrogate,
)
from active_testing_benchmark.active_testing.surrogate import (
    RandomForestModule,
    SpatialTextureFeatures,
    update_random_forest_surrogates,
)
from active_testing_benchmark.modeling import models


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
@pytest.mark.parametrize("size", [(224, 224), (73, 81), (32, 32), (128, 128)])
def test_mps_features_match_cpu_for_arbitrary_image_sizes(size):
    images = torch.randn(2, 3, *size, generator=torch.Generator().manual_seed(42))
    expected = SpatialTextureFeatures()(images)
    actual = SpatialTextureFeatures().to("mps")(images.to("mps"))
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


def test_constant_images_have_only_mean_features():
    extractor = SpatialTextureFeatures()
    features = extractor(torch.full((2, 3, 73, 81), 2.0))
    assert features.shape == (2, 960)
    offset = 0
    for grid in (4, 8):
        width = 3 * grid * grid
        torch.testing.assert_close(
            features[:, offset : offset + width], torch.full((2, width), 2.0)
        )
        torch.testing.assert_close(
            features[:, offset + width : offset + 4 * width], torch.zeros(2, 3 * width)
        )
        offset += 4 * width
    assert not list(extractor.parameters())


def test_texture_distinguishes_images_with_identical_pooled_means():
    flat = torch.zeros(1, 1, 64, 64)
    stripes = torch.ones_like(flat)
    stripes[:, :, :, ::2] = -1
    pool = nn.AdaptiveAvgPool2d((16, 16))
    torch.testing.assert_close(pool(flat), pool(stripes))
    extractor = SpatialTextureFeatures()
    assert not torch.allclose(extractor(flat), extractor(stripes))
    assert torch.isfinite(extractor(stripes)).all()
    torch.testing.assert_close(extractor(stripes), extractor(stripes))


@pytest.mark.parametrize("custom_texture", [False, True])
def test_forest_can_fit_predict_and_update_image_features(monkeypatch, custom_texture):
    build_resnet = models.resnet18
    monkeypatch.setattr(models, "resnet18", lambda *, weights: build_resnet(weights=None))
    images = torch.randn(6, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    extractor = SpatialTextureFeatures() if custom_texture else None
    forest = RandomForestModule(2, extractor, n_estimators=3).fit(
        [{"image": images, "label": labels}]
    )
    expected_type = SpatialTextureFeatures if custom_texture else models.FrozenImageNetFeatures
    assert isinstance(forest.feature_extractor, expected_type)
    assert forest.forest.n_features_in_ == (960 if custom_texture else 512)
    before = {name: value.clone() for name, value in forest.feature_extractor.named_buffers()}
    forest.train()
    predictions = forest(images).softmax(-1)
    torch.testing.assert_close(predictions.sum(-1), torch.ones(6))
    original = forest._development_features.copy()
    update_random_forest_surrogates([forest], images[:2], labels[:2])
    np.testing.assert_array_equal(forest._development_features, original)
    assert torch.isfinite(forest(images)).all()
    for name, value in forest.feature_extractor.named_buffers():
        torch.testing.assert_close(value, before[name])


class FixedLogits(nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer("logits", torch.tensor(probabilities, dtype=torch.float64).log())

    def forward(self, X):
        return self.logits.expand(len(X), -1)


def test_ensemble_averages_probabilities_and_registers_members():
    members = [FixedLogits([0.9, 0.1]), FixedLogits([0.2, 0.8])]
    ensemble = surrogate.MeanProbabilityEnsemble(members).eval()
    X = torch.zeros(3, 2)
    logits = ensemble(X)
    expected = torch.tensor([[0.55, 0.45]] * 3, dtype=torch.float64)
    torch.testing.assert_close(logits.softmax(dim=-1), expected)
    torch.testing.assert_close(logits.exp(), expected)
    assert all(not member.training for member in members)


def test_ensemble_requires_members():
    with pytest.raises(ValueError, match="at least one"):
        surrogate.MeanProbabilityEnsemble([])


def fitted_surrogates():
    development = [
        {"image": torch.tensor([0.0, 0.0]), "label": 0},
        {"image": torch.tensor([1.0, 1.0]), "label": 0},
    ]
    return tuple(
        RandomForestModule(2, nn.Identity(), n_estimators=5, random_state=seed).fit(
            DataLoader(development, batch_size=2)
        )
        for seed in (1, 2)
    )


def test_forest_callback_retains_development_data_without_repeating_old_observations(monkeypatch):
    from active_testing_benchmark.active_testing.surrogate import update_random_forest_surrogates

    models = fitted_surrogates()
    observed = torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    labels = torch.ones(3, dtype=torch.long)
    fits = [[] for _ in models]
    for model, calls in zip(models, fits):
        monkeypatch.setattr(
            model.forest, "fit", lambda X, y, calls=calls: calls.append((X.copy(), y.copy()))
        )
    update_random_forest_surrogates(models, observed[:1], labels[:1])
    update_random_forest_surrogates(models, observed, labels)
    for calls in fits:
        assert [len(y) for _, y in calls] == [3, 5]
        np.testing.assert_array_equal(calls[-1][0], [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        np.testing.assert_array_equal(calls[-1][1], [0, 0, 1, 1, 1])
