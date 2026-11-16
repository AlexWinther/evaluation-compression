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
        ppat_proxy_correlation: Pearson correlation of the proxy loss
            L(f(x), argmax proxy(x)) with the true loss.
    """
    model.eval()
    logits, surrogate_probabilities, proxy_predictions, labels = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images = batch["image"].to(device)
        logits.append(model(images).cpu())
        # Softmax turns both logits and log-probabilities (the forests' outputs) into
        # probabilities. Members are averaged as in MeanProbabilityEnsemble.
        member_probabilities = [member(images).softmax(dim=1).cpu() for member in surrogates]
        surrogate_probabilities.append(torch.stack(member_probabilities).mean(dim=0))
        proxy_predictions.append(proxy(images).argmax(dim=1).cpu())
        labels.append(torch.as_tensor(batch["label"]).long())

    # class_losses[i, c] is the cross-entropy of item i if its label were class c.
    class_losses = -torch.cat(logits).double().log_softmax(dim=1).numpy()
    rows = np.arange(len(class_losses))
    losses = class_losses[rows, torch.cat(labels).numpy()]
    expected_losses = (torch.cat(surrogate_probabilities).double().numpy() * class_losses).sum(1)
    proxy_losses = class_losses[rows, torch.cat(proxy_predictions).numpy()]
    # The predicted class has the smallest hypothetical loss.
    model_errors = class_losses.argmin(axis=1) != torch.cat(labels).numpy()
    error_auc = (
        roc_auc_score(model_errors, expected_losses)
        if 0 < model_errors.sum() < len(rows)
        else None
    )

    # The model's own expected loss, sum_c p_f(c|x) loss(c), is its predictive entropy.
    target_expected_losses = (np.exp(-class_losses) * class_losses).sum(1)
    confidence_losses = np.sqrt(confidence_binned_losses(class_losses, losses**2))

    risk = losses.mean()
    return {
        "kossen_score_spearman": _finite_or_none(spearmanr(expected_losses, losses)[0]),
        "kossen_error_auc": error_auc,
        "kossen_variance_gain": variance_gain(expected_losses, losses),
        "oracle_variance_gain": variance_gain(losses, losses),
        "target_variance_gain": variance_gain(target_expected_losses, losses),
        "confidence_oracle_variance_gain": variance_gain(confidence_losses, losses),
        "ase_relative_bias": float((expected_losses.mean() - risk) / risk),
        "ppat_proxy_correlation": _finite_or_none(np.corrcoef(proxy_losses, losses)[0, 1]),
    }
