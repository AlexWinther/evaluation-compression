"""Check whether the surrogates and proxy can help before running the benchmark.

Active testing only beats uniform sampling when its acquisition scores track the
true per-item loss (Kossen et al., 2021, Sec. 3). PPAT additionally needs proxy
losses that correlate with the true losses (Ashouritaklimi et al., 2026, Prop. 3.2).
These diagnostics measure both on a labelled holdout that is never the evaluation
pool, so they can be reported without touching evaluation labels.
"""

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from active_testing_benchmark.active_testing.surrogate import LabelOracle


def variance_gain(scores: np.ndarray, losses: np.ndarray, exploration: float = 0.2) -> float:
    """Return Var(uniform) / Var(importance sampling) for one draw from the proposal.

    The proposal is proportional to `scores`, mixed with uniform sampling as in
    `KossenActiveTesting`. Values above 1 mean the proposal needs fewer labels than
    uniform sampling for the same variance. Proposal q gives the one-draw estimate
    loss_i / (N q_i), whose variance is sum_i q_i (loss_i / (N q_i) - risk)^2.
    """
    N = len(losses)
    risk = losses.mean()
    q = (1 - exploration) * scores / scores.sum() + exploration / N
    importance_variance = np.sum(q * (losses / (N * q) - risk) ** 2)
    uniform_variance = np.mean((losses - risk) ** 2)
    return float(uniform_variance / importance_variance)


def confidence_binned_losses(
    class_losses: np.ndarray, losses: np.ndarray, n_bins: int = 10
) -> np.ndarray:
    """Replace each item's value by the mean value of items with similar model confidence.

    Items are grouped by predicted class, then into `n_bins` quantile bins of the
    model's probability for that class. Every proposal that depends only on the
    model's own output, such as the target model's expected loss or any
    recalibration of it (Lee et al., 2024), is a function of these two quantities.

    Pass squared losses and take the square root to get the best such proposal.
    One importance-sampling draw has variance sum_i loss_i^2 / (N^2 q_i) - risk^2.
    Minimising it over proposals that are constant within each bin gives
    q proportional to sqrt(E[loss^2 | bin]), not to E[loss | bin]. Uniform sampling
    is one such proposal, so the best one is never worse than uniform.

    This is an in-sample estimate, not a strict limit. The bin values use the
    holdout's true losses, which no real proposal can see, so the estimate is
    optimistic for the given bins. Finer bins give larger in-sample values, but
    those come from bins that happen to hold no errors and do not carry over to new
    images. With rare errors, fitting the bins on one half of the holdout and
    scoring the other half gives lower values, and fewer bins do better there.
    """
    predicted_classes = class_losses.argmin(axis=1)
    # The predicted class has the smallest hypothetical loss, exp(-loss) is its probability.
    confidences = np.exp(-class_losses.min(axis=1))
    binned = np.empty_like(losses)
    for predicted_class in np.unique(predicted_classes):
        in_class = np.flatnonzero(predicted_classes == predicted_class)
        # Rank-based bins keep their sizes equal even when many confidences tie.
        ranks = np.argsort(np.argsort(confidences[in_class], kind="stable"), kind="stable")
        bins = ranks * min(n_bins, len(in_class)) // len(in_class)
        for bin_index in np.unique(bins):
            members = in_class[bins == bin_index]
            binned[members] = losses[members].mean()
    return binned


def _finite_or_none(value: float) -> float | None:
    """JSON metadata rejects NaN, which a constant input gives for correlations."""
    return float(value) if math.isfinite(value) else None


def proxy_diagnostics(
    class_losses: np.ndarray, labels: np.ndarray, proxy_probabilities: np.ndarray
) -> dict[str, float | None]:
    """Measure how useful a PPAT proxy is for one model, from held-out labelled data.

    `class_losses[i, c]` is the model's loss if item i had label c, and
    `proxy_probabilities` are the proxy's class probabilities. PPAT uses the hard
    proxy label, so its proxy loss is L(f(x), argmax g(x)) (Ashouritaklimi et al., 2026,
    p.4). The soft proxy loss sum_c g(c|x) L(f(x), c) is reported as an ablation.

    Returns:
        ppat_proxy_correlation: Pearson correlation of the hard proxy loss with the
            true loss. PPAT with lambda = 1/2 beats LURE when this exceeds about 1/4,
            if both losses have similar spread (Prop. 3.2 and Sec. 3.3).
        ppat_soft_proxy_correlation: the same for the soft proxy loss.
        ppat_lambda_dagger: the proposal-free coefficient sum(l c) / sum(c^2) with
            c the centred hard proxy loss (Prop. 3.4). Fixed lambda = 1/2 reduces
            variance when it lies in (0, 2 lambda), that is, when this exceeds 1/4.
            It is a check, not a value to tune lambda with.
        proxy_auc: AUC of the proxy for the labels (binary or one-vs-rest macro).
        proxy_accuracy: accuracy of the hard proxy labels.
        proxy_class_frequency_error: largest gap between a class's hard proxy
            frequency and its true frequency. Large values mean the hard labels
            ignore a class, as when a proxy never predicts the rare positive class.
    """
    rows = np.arange(len(labels))
    losses = class_losses[rows, labels]
    hard_labels = proxy_probabilities.argmax(axis=1)
    hard_losses = class_losses[rows, hard_labels]
    soft_losses = (proxy_probabilities * class_losses).sum(axis=1)
    centred = hard_losses - hard_losses.mean()
    num_classes = class_losses.shape[1]
    return {
        "ppat_proxy_correlation": _correlation(hard_losses, losses),
        "ppat_soft_proxy_correlation": _correlation(soft_losses, losses),
        "ppat_lambda_dagger": (
            float(np.sum(losses * centred) / np.sum(centred**2)) if np.any(centred) else None
        ),
        "proxy_auc": _auc(labels, proxy_probabilities),
        "proxy_accuracy": float((hard_labels == labels).mean()),
        "proxy_class_frequency_error": float(
            np.abs(
                np.bincount(hard_labels, minlength=num_classes)
                - np.bincount(labels, minlength=num_classes)
            ).max()
            / len(labels)
        ),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Pearson correlation, or None when either input is constant."""
    if np.std(first) == 0 or np.std(second) == 0:
        return None
    return _finite_or_none(np.corrcoef(first, second)[0, 1])


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    """Binary or one-vs-rest macro AUC, or None when a class is missing."""
    num_classes = probabilities.shape[1]
    if not np.array_equal(np.unique(labels), np.arange(num_classes)):
        return None
    if num_classes == 2:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))


def xwed_scores(member_probabilities: np.ndarray, class_losses: np.ndarray) -> np.ndarray:
    """XWED of Kossen et al. (2022, Eq. 6) for [members, items, classes] probabilities."""
    mean = member_probabilities.mean(axis=0)
    member_p_log_p = np.where(
        member_probabilities > 0,
        member_probabilities * np.log(np.maximum(member_probabilities, 1e-300)),
        0.0,
    ).mean(axis=0)
    mean_p_log_p = mean * np.log(np.maximum(mean, 1e-300))
    return np.maximum(((member_p_log_p - mean_p_log_p) * class_losses).sum(axis=1), 0.0)


@torch.inference_mode()
def surrogate_diagnostics(
    model: nn.Module,
    surrogates: tuple[nn.Module, ...],
    proxy: nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    batch_size: int = 64,
) -> dict[str, float | None]:
    """Measure, for cross-entropy, how well the auxiliary models track the model's losses.

    Returns:
        kossen_score_spearman: rank correlation of the surrogate-expected loss
            (the Kossen acquisition score, Eq. 12) with the true loss.
        kossen_error_auc: AUC of the Kossen score for separating the images the
            model misclassifies from the rest. Unlike the surrogate's own AUC, this
            measures whether it finds the model's errors, which hold most of the loss.
            None when the model makes no errors, or only errors.
        kossen_variance_gain: `variance_gain` of the Kossen proposal.
        oracle_variance_gain: `variance_gain` of a proposal proportional to the
            true loss. This is the upper limit for this model and holdout.
        target_variance_gain: `variance_gain` of a proposal that uses the model
            itself as surrogate, that is, its own expected loss (predictive
            entropy). This is Kossen without a surrogate, or "Ori" in Lee et al.
        confidence_oracle_variance_gain: `variance_gain` of the best proposal
            that depends only on the model's confidence, from
            `confidence_binned_losses` on the squared losses, fitted and scored
            on the same holdout. It is at least about 1 (uniform is in the
            family) and optimistic, not a strict upper limit. If it is near 1,
            surrogate-free proposals and recalibrations are unlikely to
            meaningfully beat uniform sampling.
        ase_relative_bias: relative error of the ASE estimate before any labels,
            (mean expected loss - true risk) / true risk.
        surrogate_nll: negative log-likelihood of the labels under the ensemble's
            mean probabilities. Lower means a better calibrated, sharper surrogate.
        surrogate_class_frequency_error: largest gap between the ensemble's mean
            probability of a class and its true frequency (calibration in the
            large). It is the direct cause of ASE bias on rare classes.
        member_disagreement: mean fraction of items on which two members predict
            different classes. None for a single member.
        xwed_spearman: rank correlation of the XWED score with the true loss.
            None for a single member. XWED ranks where a label would most change
            the surrogate's loss-weighted prediction, which for a confident model is
            often on its confident, correct images. It is not meant to track the
            loss, so a low or negative value is not by itself a failure of ASE.
        Plus every entry of `proxy_diagnostics` for the proxy.
    """
    model.eval()
    # A LabelOracle control predicts from the labels, not from the images, so its
    # outputs are computed after the loop from the collected labels and logits.
    # Keyed by identity, so a module that is both a surrogate and the proxy runs once.
    image_modules = {
        id(m): m for m in (*surrogates, proxy) if not isinstance(m, LabelOracle)
    }.values()
    logits, labels = [], []
    outputs: dict[int, list[torch.Tensor]] = {id(module): [] for module in image_modules}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = batch["image"].to(device)
        logits.append(model(images).cpu())
        for module in image_modules:
            outputs[id(module)].append(module(images).cpu())
        labels.append(torch.as_tensor(batch["label"]).long())

    logits = torch.cat(logits).double()
    labels = torch.cat(labels).numpy()

    def probabilities(module: nn.Module) -> np.ndarray:
        # Softmax turns both logits and log-probabilities (the forests' outputs) into
        # probabilities. Members are averaged as in MeanProbabilityEnsemble.
        if isinstance(module, LabelOracle):
            module_outputs = torch.from_numpy(module.log_probabilities(labels, logits.numpy()))
        else:
            module_outputs = torch.cat(outputs[id(module)])
        return module_outputs.double().softmax(dim=1).numpy()

    # class_losses[i, c] is the cross-entropy of item i if its label were class c.
    class_losses = -logits.log_softmax(dim=1).numpy()
    rows = np.arange(len(class_losses))
    losses = class_losses[rows, labels]
    members = np.stack([probabilities(member) for member in surrogates])
    surrogate_probabilities = members.mean(axis=0)
    expected_losses = (surrogate_probabilities * class_losses).sum(1)
    # The predicted class has the smallest hypothetical loss.
    model_errors = class_losses.argmin(axis=1) != labels
    error_auc = (
        roc_auc_score(model_errors, expected_losses)
        if 0 < model_errors.sum() < len(rows)
        else None
    )

    # The model's own expected loss, sum_c p_f(c|x) loss(c), is its predictive entropy.
    target_expected_losses = (np.exp(-class_losses) * class_losses).sum(1)
    confidence_losses = np.sqrt(confidence_binned_losses(class_losses, losses**2))

    num_classes = class_losses.shape[1]
    member_disagreement, xwed_spearman = None, None
    if len(members) > 1:
        predictions = members.argmax(axis=2)
        member_disagreement = float(
            np.mean(
                [
                    (predictions[first] != predictions[second]).mean()
                    for first in range(len(members))
                    for second in range(first + 1, len(members))
                ]
            )
        )
        xwed_spearman = _finite_or_none(spearmanr(xwed_scores(members, class_losses), losses)[0])

    risk = losses.mean()
    return {
        "kossen_score_spearman": _finite_or_none(spearmanr(expected_losses, losses)[0]),
        "kossen_error_auc": error_auc,
        "kossen_variance_gain": variance_gain(expected_losses, losses),
        "oracle_variance_gain": variance_gain(losses, losses),
        "target_variance_gain": variance_gain(target_expected_losses, losses),
        "confidence_oracle_variance_gain": variance_gain(confidence_losses, losses),
        "ase_relative_bias": float((expected_losses.mean() - risk) / risk),
        "surrogate_nll": float(
            -np.log(np.maximum(surrogate_probabilities[rows, labels], 1e-300)).mean()
        ),
        "surrogate_class_frequency_error": float(
            np.abs(
                surrogate_probabilities.mean(axis=0)
                - np.bincount(labels, minlength=num_classes) / len(labels)
            ).max()
        ),
        "member_disagreement": member_disagreement,
        "xwed_spearman": xwed_spearman,
        **proxy_diagnostics(class_losses, labels, probabilities(proxy)),
    }
