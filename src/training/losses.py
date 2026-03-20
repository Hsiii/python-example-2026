from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only when torch is unavailable.
    torch = None


def cox_ph_loss(log_risk: "torch.Tensor", durations: "torch.Tensor", events: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise ImportError("PyTorch is required for cox_ph_loss.")

    durations = durations.reshape(-1)
    events = events.reshape(-1)
    log_risk = log_risk.reshape(-1)

    order = torch.argsort(durations, descending=True)
    sorted_log_risk = log_risk[order]
    sorted_events = events[order]
    log_cumulative_hazard = torch.logcumsumexp(sorted_log_risk, dim=0)
    observed = sorted_log_risk - log_cumulative_hazard
    loss = -(observed * sorted_events).sum() / torch.clamp(sorted_events.sum(), min=1.0)
    return loss


def make_time_bins(durations: np.ndarray, num_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    raw_bins = np.quantile(durations, quantiles)
    bins = np.unique(raw_bins.astype(np.float64))
    if bins.size == 0:
        bins = np.asarray([float(np.max(durations))], dtype=np.float64)
    return bins


def discretize_durations(durations: np.ndarray, time_bins: np.ndarray) -> np.ndarray:
    return np.digitize(durations, time_bins, right=True).astype(np.int64)


def discrete_time_nll(
    logits: "torch.Tensor",
    bin_index: "torch.Tensor",
    events: "torch.Tensor",
) -> "torch.Tensor":
    if torch is None:
        raise ImportError("PyTorch is required for discrete_time_nll.")

    hazards = torch.sigmoid(logits)
    log_hazards = torch.log(torch.clamp(hazards, min=1e-8, max=1.0))
    log_survival = torch.log(torch.clamp(1.0 - hazards, min=1e-8, max=1.0))
    cumulative_log_survival = torch.cumsum(log_survival, dim=1)

    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    bin_index = bin_index.long().clamp(min=0, max=logits.shape[1] - 1)
    previous_bin = torch.clamp(bin_index - 1, min=0)

    survival_before_event = torch.where(
        bin_index > 0,
        cumulative_log_survival[batch_indices, previous_bin],
        torch.zeros_like(events),
    )
    event_log_likelihood = survival_before_event + log_hazards[batch_indices, bin_index]
    censor_log_likelihood = cumulative_log_survival[batch_indices, bin_index]
    log_likelihood = torch.where(events > 0.5, event_log_likelihood, censor_log_likelihood)
    return -log_likelihood.mean()


def deephit_ranking_loss(
    logits: "torch.Tensor",
    durations: "torch.Tensor",
    events: "torch.Tensor",
) -> "torch.Tensor":
    if torch is None:
        raise ImportError("PyTorch is required for deephit_ranking_loss.")

    with torch.no_grad():
        order = durations.reshape(-1, 1) < durations.reshape(1, -1)
        comparable = order & (events.reshape(-1, 1) > 0.5)

    cumulative_incidence = 1.0 - torch.cumprod(1.0 - torch.sigmoid(logits), dim=1)
    risk_score = cumulative_incidence.mean(dim=1)
    margin = risk_score.reshape(-1, 1) - risk_score.reshape(1, -1)
    ranking_penalty = torch.exp(-margin)
    valid_penalty = ranking_penalty[comparable]
    if valid_penalty.numel() == 0:
        return torch.zeros((), device=logits.device)
    return valid_penalty.mean()
