import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from active_testing_benchmark.active_testing import (
    surrogate,
)
from active_testing_benchmark.active_testing.diagnostics import (
    confidence_binned_losses,
    surrogate_diagnostics,
    variance_gain,
)
from active_testing_benchmark.active_testing.surrogate import (
    ConstantClassSurrogate,
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


def test_variance_gain_is_one_for_uniform_scores_and_large_for_the_true_loss():
    losses = np.array([0.01, 0.02, 0.05, 3.0, 0.1, 4.0])
    assert variance_gain(np.ones_like(losses), losses) == pytest.approx(1.0)
    assert variance_gain(losses, losses) > 10
    assert variance_gain(1 / losses, losses) < 1


def hypothetical_losses(positive_probabilities: list[float]) -> np.ndarray:
    """Return binary cross-entropy for each item if its label were class 0 or class 1."""
    p = np.array(positive_probabilities)
    return -np.log(np.column_stack([1 - p, p]))


def test_confidence_binning_cannot_separate_errors_at_the_same_confidence():
    # Four items predicted class 1 with 0.9 confidence, one of them wrongly.
    class_losses = hypothetical_losses([0.9, 0.9, 0.9, 0.9])
    losses = class_losses[np.arange(4), [1, 1, 1, 0]]
    binned = confidence_binned_losses(class_losses, losses, n_bins=1)
    np.testing.assert_allclose(binned, np.full(4, losses.mean()))
    # Equal scores give the uniform proposal, so the confident error stays hidden.
    assert variance_gain(binned, losses) == pytest.approx(1.0)
    assert variance_gain(losses, losses) > 1


def test_rms_confidence_proposal_is_never_worse_than_uniform_but_mean_can_be():
    # Eight confident predictions with one confident error, and eight moderate errors.
    class_losses = hypothetical_losses([0.99] * 8 + [0.6] * 8)
    labels = [1] * 7 + [0] + [0] * 8
    losses = class_losses[np.arange(16), labels]
    mean_proposal = confidence_binned_losses(class_losses, losses, n_bins=2)
    rms_proposal = np.sqrt(confidence_binned_losses(class_losses, losses**2, n_bins=2))
    # The mean undersamples the confident bin, which holds the largest single loss.
    assert variance_gain(mean_proposal, losses) < 1
    assert variance_gain(rms_proposal, losses) > 1


def test_confidence_binning_keeps_losses_that_confidence_explains():
    # Each confidence level, per predicted class, has its own bin and a single loss.
    class_losses = hypothetical_losses([0.95, 0.7, 0.55, 0.4, 0.2, 0.05])
    labels = [1, 1, 0, 0, 0, 1]
    losses = class_losses[np.arange(6), labels]
    binned = confidence_binned_losses(class_losses, losses, n_bins=3)
    np.testing.assert_allclose(binned, losses)


class Scaled(nn.Module):
    """Return confident logits that put almost all mass on the input's one-hot class."""

    def forward(self, images):
        return 20 * images


class Inverted(nn.Module):
    """Return confident logits for the class opposite to the input's one-hot class."""

    def forward(self, images):
        return 20 * (1 - images)


def test_surrogate_diagnostics_separate_informative_and_uninformative_surrogates():
    labels = [0, 1, 0, 1, 1, 0, 0, 1]
    images = torch.eye(2)[labels]
    dataset = [{"image": image, "label": label} for image, label in zip(images, labels)]
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    # The constant surrogate ignores its input. The scaled identity knows the label.
    constant = nn.Linear(2, 2)
    nn.init.zeros_(constant.weight)

    informative = surrogate_diagnostics(model, (Scaled(),), Scaled(), dataset, "cpu", 3)
    uninformative = surrogate_diagnostics(model, (constant,), constant, dataset, "cpu", 3)

    assert informative["kossen_variance_gain"] > uninformative["kossen_variance_gain"]
    # A surrogate that predicts the wrong label scores the model's correct images highest.
    inverted = surrogate_diagnostics(model, (Inverted(),), Scaled(), dataset, "cpu", 3)
    assert informative["kossen_error_auc"] == pytest.approx(1.0)
    assert inverted["kossen_error_auc"] == pytest.approx(0.0)
    assert informative["kossen_variance_gain"] == pytest.approx(
        informative["oracle_variance_gain"], rel=1e-3
    )
    assert informative["ase_relative_bias"] == pytest.approx(0.0, abs=1e-6)
    # The surrogate-free diagnostics depend only on the model, not on the surrogates.
    for name in ("target_variance_gain", "confidence_oracle_variance_gain"):
        assert informative[name] == pytest.approx(uninformative[name])
        assert informative[name] > 0
    # A proxy that predicts the true label has proxy losses equal to the true losses.
    assert informative["ppat_proxy_correlation"] == pytest.approx(1.0)


def test_constant_class_surrogate_scores_the_loss_of_its_class():
    surrogate = ConstantClassSurrogate(num_classes=2, target_class=1)
    images = torch.randn(4, 2)
    probabilities = surrogate(images).softmax(dim=1)
    torch.testing.assert_close(probabilities, torch.tensor([[0.0, 1.0]] * 4), atol=1e-6, rtol=0)

    # The model is sure of class 0 for the first two items and of class 1 for the rest.
    logits = torch.tensor([[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]])
    expected_losses = (probabilities * -logits.log_softmax(dim=1)).sum(dim=1)
    torch.testing.assert_close(expected_losses, -logits.log_softmax(dim=1)[:, 1], atol=1e-5, rtol=0)
    assert expected_losses[0] > expected_losses[2]


def test_constant_class_surrogate_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="target_class"):
        ConstantClassSurrogate(num_classes=2, target_class=2)
