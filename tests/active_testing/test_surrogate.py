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
    proxy_diagnostics,
    surrogate_diagnostics,
    variance_gain,
    xwed_scores,
)
from active_testing_benchmark.active_testing.surrogate import (
    ConstantClassSurrogate,
    LabelOracle,
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


def test_temperature_fit_recovers_the_temperature_that_generated_the_labels():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=4.0, size=(20000, 3))
    true_temperature = 2.5
    probabilities = np.exp(logits / true_temperature)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=row) for row in probabilities])
    assert surrogate.fit_temperature(logits, labels) == pytest.approx(true_temperature, rel=0.05)


def test_temperature_scaling_divides_logits_and_keeps_the_predicted_class():
    model = FixedLogits([0.7, 0.2, 0.1])
    scaled = surrogate.TemperatureScaled(model, temperature=2.0)
    inputs = torch.zeros(4, 1)
    torch.testing.assert_close(scaled(inputs), model(inputs) / 2.0)
    assert torch.equal(scaled(inputs).argmax(dim=1), model(inputs).argmax(dim=1))
    with pytest.raises(ValueError, match="positive"):
        surrogate.TemperatureScaled(model, temperature=0.0)


def separable_development(count=40):
    """Two-dimensional features whose first coordinate decides an imbalanced label."""
    rng = np.random.default_rng(3)
    features = torch.tensor(rng.normal(size=(count, 2)), dtype=torch.float32)
    labels = (features[:, 0] > 1.0).long()
    return [{"image": x, "label": y} for x, y in zip(features, labels)]


def test_forest_models_the_development_prevalence_without_class_weights():
    forest = RandomForestModule(2, nn.Identity(), n_estimators=5)
    assert forest.forest.class_weight is None
    assert forest.forest.criterion == "entropy"
    assert forest.forest.max_features == "sqrt"


def test_bootstrapped_forests_keep_distinct_resamples_across_refits(monkeypatch):
    development = separable_development()
    loader = DataLoader(development, batch_size=8)
    original = torch.stack([item["image"] for item in development]).numpy()
    forests = [
        RandomForestModule(
            2, nn.Identity(), n_estimators=5, random_state=seed, bootstrap_development=True
        ).fit(loader)
        for seed in (1, 2)
    ]
    first, second = (forest._development_features for forest in forests)
    assert first.shape == second.shape == original.shape
    assert not np.array_equal(first, second)
    # Every resampled row is a development row, and some rows repeat.
    assert all((original == row).all(axis=1).any() for row in first)
    assert len(np.unique(first, axis=0)) < len(first)
    again = RandomForestModule(
        2, nn.Identity(), n_estimators=5, random_state=1, bootstrap_development=True
    ).fit(loader)
    np.testing.assert_array_equal(again._development_features, first)

    fits = []
    monkeypatch.setattr(forests[0].forest, "fit", lambda X, y: fits.append((X.copy(), y.copy())))
    observed = torch.tensor([[5.0, 5.0], [6.0, 6.0]])
    update_random_forest_surrogates(forests[:1], observed, torch.ones(2, dtype=torch.long))
    np.testing.assert_array_equal(fits[0][0][: len(first)], first)
    np.testing.assert_array_equal(fits[0][0][len(first) :], observed.numpy())


def test_forest_calibration_sets_a_temperature_and_keeps_valid_probabilities():
    development = separable_development()
    forest = RandomForestModule(2, nn.Identity(), n_estimators=10, random_state=0).fit(
        DataLoader(development[:30], batch_size=8)
    )
    images = torch.stack([item["image"] for item in development[30:]])
    uncalibrated = forest(images).softmax(dim=1)
    forest.calibrate(DataLoader(development[30:], batch_size=8))
    assert forest.temperature > 0
    calibrated = forest(images).softmax(dim=1)
    torch.testing.assert_close(calibrated.sum(dim=1), torch.ones(len(images)))
    assert torch.equal(calibrated.argmax(dim=1), uncalibrated.argmax(dim=1))


def test_frozen_feature_probe_returns_log_probabilities_for_every_class():
    development = separable_development()
    probe = surrogate.FrozenFeatureProbe(3, nn.Identity()).fit(DataLoader(development, 8))
    images = torch.stack([item["image"] for item in development])
    log_probabilities = probe(images)
    assert log_probabilities.shape == (len(images), 3)
    # Class 2 never occurs, so it gets the probability floor.
    torch.testing.assert_close(log_probabilities.exp().sum(dim=1), torch.ones(len(images)))
    assert (log_probabilities[:, 2] < -15).all()
    labels = torch.stack([item["label"] for item in development])
    assert (log_probabilities.argmax(dim=1) == labels).float().mean() > 0.9


def test_proxy_diagnostics_reward_a_proxy_that_knows_the_labels():
    labels = np.array([0, 0, 0, 1, 0, 1, 0, 0])
    # A model that is confident in class 0 everywhere, so it misses both positives.
    class_losses = np.tile(-np.log([0.9, 0.1]), (len(labels), 1))
    class_losses[::2] = -np.log([0.7, 0.3])
    oracle = np.eye(2)[labels]
    perfect = proxy_diagnostics(class_losses, labels, oracle)
    assert perfect["ppat_proxy_correlation"] == pytest.approx(1.0)
    assert perfect["ppat_lambda_dagger"] == pytest.approx(1.0)
    assert perfect["proxy_accuracy"] == 1.0
    assert perfect["proxy_class_frequency_error"] == 0.0
    # A proxy that never predicts the rare class misses its whole frequency.
    negative = proxy_diagnostics(class_losses, labels, np.tile([0.8, 0.2], (len(labels), 1)))
    assert negative["proxy_class_frequency_error"] == pytest.approx(labels.mean())
    assert negative["proxy_auc"] == pytest.approx(0.5)
    assert negative["ppat_soft_proxy_correlation"] is not None


def test_xwed_is_zero_when_members_agree_and_grows_with_disagreement():
    class_losses = np.array([[0.1, 2.0], [0.1, 2.0]])
    agree = np.array([[[0.8, 0.2], [0.8, 0.2]], [[0.8, 0.2], [0.8, 0.2]]])
    disagree = np.array([[[0.8, 0.2], [0.99, 0.01]], [[0.8, 0.2], [0.5, 0.5]]])
    np.testing.assert_allclose(xwed_scores(agree, class_losses), 0.0, atol=1e-12)
    scores = xwed_scores(disagree, class_losses)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] > 0


def test_surrogate_diagnostics_report_calibration_and_member_diversity():
    labels = [0, 1, 0, 1, 1, 0, 0, 1]
    images = torch.eye(2)[labels]
    dataset = [{"image": image, "label": label} for image, label in zip(images, labels)]
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    single = surrogate_diagnostics(model, (Scaled(),), Scaled(), dataset, "cpu", 3)
    assert single["member_disagreement"] is None
    assert single["xwed_spearman"] is None
    assert single["surrogate_class_frequency_error"] == pytest.approx(0.0, abs=1e-6)
    pair = surrogate_diagnostics(model, (Scaled(), Inverted()), Scaled(), dataset, "cpu", 3)
    assert pair["member_disagreement"] == pytest.approx(1.0)
    assert pair["xwed_spearman"] is not None
    assert pair["surrogate_nll"] > single["surrogate_nll"]
    assert pair["ppat_lambda_dagger"] == pytest.approx(1.0)


def test_exact_oracle_returns_the_true_labels_and_ignores_noise():
    labels = np.array([0, 1, 1, 0, 2])
    logits = np.zeros((5, 3))
    oracle = LabelOracle(3, "oracle", noise=0.9, seed=1)
    probabilities = np.exp(oracle.log_probabilities(labels, logits))
    np.testing.assert_allclose(probabilities, np.eye(3)[labels], atol=1e-7)
    with pytest.raises(RuntimeError, match="log_probabilities"):
        oracle(torch.zeros(5, 3))


def test_noisy_oracle_flips_labels_and_gives_surrogates_the_calibrated_posterior():
    rng = np.random.default_rng(0)
    labels = (rng.random(20_000) < 0.1).astype(int)
    logits = np.zeros((len(labels), 2))
    proxy = LabelOracle(2, "noisy_oracle", noise=0.2, seed=5, hard_labels=True)
    noisy_labels = proxy.log_probabilities(labels, logits).argmax(axis=1)
    assert (noisy_labels != labels).mean() == pytest.approx(0.2, abs=0.01)
    # The same seed gives the same grader, and another seed errs on other items.
    np.testing.assert_array_equal(
        noisy_labels, proxy.log_probabilities(labels, logits).argmax(axis=1)
    )
    other = LabelOracle(2, "noisy_oracle", noise=0.2, seed=6, hard_labels=True)
    assert (other.log_probabilities(labels, logits).argmax(axis=1) != noisy_labels).any()

    surrogate_oracle = LabelOracle(2, "noisy_oracle", noise=0.2, seed=5)
    positive = np.exp(surrogate_oracle.log_probabilities(labels, logits))[:, 1]
    # Calibrated: among the items with one posterior value, that share is positive.
    for value in np.unique(positive):
        assert labels[positive == value].mean() == pytest.approx(value, abs=0.02)
    # With a rare positive class, a positive call from this grader is right less than
    # half of the time, so the posterior differs from the hard label.
    assert positive.max() < 0.5


def test_model_noisy_oracle_interpolates_between_the_labels_and_the_model():
    labels = np.array([0, 1] * 500)
    logits = np.tile([[2.0, -1.0]], (1000, 1))
    model_probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    exact = np.exp(LabelOracle(2, "model_noisy_oracle", 0.0).log_probabilities(labels, logits))
    np.testing.assert_allclose(exact, np.eye(2)[labels], atol=1e-7)
    model_only = LabelOracle(2, "model_noisy_oracle", 1.0).log_probabilities(labels, logits)
    np.testing.assert_allclose(np.exp(model_only), model_probabilities, rtol=1e-5)
    half = np.exp(LabelOracle(2, "model_noisy_oracle", 0.5).log_probabilities(labels, logits))
    from_model = np.isclose(half, model_probabilities, rtol=1e-5).all(axis=1)
    assert from_model.mean() == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [((2, "gp"), "kind"), ((2, "noisy_oracle", 1.5), "noise"), ((1, "oracle"), "two classes")],
)
def test_label_oracle_rejects_invalid_settings(arguments, message):
    with pytest.raises(ValueError, match=message):
        LabelOracle(*arguments)


def test_surrogate_diagnostics_reach_the_ceiling_with_an_oracle_and_fall_with_noise():
    rng = np.random.default_rng(0)
    labels = (rng.random(400) < 0.3).astype(int)
    images = torch.from_numpy(rng.normal(size=(400, 2))).float()
    dataset = [{"image": image, "label": int(label)} for image, label in zip(images, labels)]
    model = nn.Identity()

    def diagnostics(kind, noise):
        return surrogate_diagnostics(
            model,
            (LabelOracle(2, kind, noise, seed=1),),
            LabelOracle(2, kind, noise, seed=2, hard_labels=True),
            dataset,
            "cpu",
            64,
        )

    exact = diagnostics("oracle", 0.0)
    assert exact["kossen_variance_gain"] == pytest.approx(exact["oracle_variance_gain"], rel=1e-3)
    assert exact["kossen_score_spearman"] == pytest.approx(1.0)
    assert exact["ppat_proxy_correlation"] == pytest.approx(1.0)
    assert exact["ase_relative_bias"] == pytest.approx(0.0, abs=1e-6)
    for kind in ("noisy_oracle", "model_noisy_oracle"):
        noisy = diagnostics(kind, 0.3)
        assert noisy["kossen_score_spearman"] < 1.0
        assert noisy["kossen_variance_gain"] < exact["kossen_variance_gain"]
        assert noisy["ppat_proxy_correlation"] < 1.0
    # At full noise the correlated oracle is the model used as its own surrogate.
    model_only = diagnostics("model_noisy_oracle", 1.0)
    assert model_only["kossen_variance_gain"] == pytest.approx(
        model_only["target_variance_gain"], rel=1e-4
    )
