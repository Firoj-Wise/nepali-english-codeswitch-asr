"""Group-relative policy optimization objectives.

Supported variants, selected by GrpoObjectiveSettings:
- GRPO: group std normalization, sequence-mean aggregation, token ratio, KL to reference.
- DAPO: asymmetric clip (clip_low < clip_high), token-mean aggregation, dynamic sampling
  (drop groups whose rewards are all equal), typically zero KL.
- Dr. GRPO: no std normalization, constant normalizer aggregation.
- GSPO: sequence-level, length-normalized importance ratio with sequence-level clipping.

The importance ratio is taken against the log probabilities recorded when the samples were
drawn (old_log_probs). With one update epoch the ratio is exactly one and clipping has no
effect; clipping matters only when update_epochs is greater than one.
"""

from dataclasses import dataclass

import torch

from csrl.domain import constants
from csrl.domain.enums import AdvantageNormalization, ImportanceRatio, LossAggregation
from csrl.domain.errors import CsrlError


@dataclass(frozen=True)
class GrpoObjectiveSettings:
    advantage_normalization: AdvantageNormalization
    loss_aggregation: LossAggregation
    importance_ratio: ImportanceRatio
    clip_low: float
    clip_high: float
    kl_coefficient: float
    constant_normalizer: int


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    mean_kl: float
    clipped_fraction: float


def group_advantages(
    rewards: torch.Tensor, group_size: int, normalization: AdvantageNormalization
) -> torch.Tensor:
    """Advantages [N] for rewards [N] laid out as consecutive groups of group_size."""
    if rewards.numel() % group_size != 0:
        raise CsrlError(f"{rewards.numel()} rewards do not divide into groups of {group_size}")
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalization is AdvantageNormalization.GROUP_STD:
        centered = centered / (grouped.std(dim=1, unbiased=False, keepdim=True)
                               + constants.ADVANTAGE_EPSILON)
    return centered.flatten()


def nonconstant_group_mask(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Row mask [N]: True for samples whose group has at least two distinct rewards."""
    grouped = rewards.view(-1, group_size)
    varied = grouped.max(dim=1).values > grouped.min(dim=1).values
    return varied.repeat_interleave(group_size)


def _kl_k3(log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
    """Per-token k3 estimator of KL(policy || reference): exp(d) - d - 1 with d = ref - policy."""
    difference = reference_log_probs - log_probs
    return torch.exp(difference) - difference - 1


def _aggregate(
    per_token: torch.Tensor, mask: torch.Tensor, settings: GrpoObjectiveSettings
) -> torch.Tensor:
    weights = mask.float()
    lengths = weights.sum(dim=1)
    if bool((lengths == 0).any()):
        raise CsrlError("a sequence has no action tokens; generation must emit at least one")
    if settings.loss_aggregation is LossAggregation.SEQUENCE_MEAN:
        return ((per_token * weights).sum(dim=1) / lengths).mean()
    if settings.loss_aggregation is LossAggregation.TOKEN_MEAN:
        return (per_token * weights).sum() / weights.sum()
    if settings.loss_aggregation is LossAggregation.CONSTANT_NORMALIZER:
        return (per_token * weights).sum() / (per_token.shape[0] * settings.constant_normalizer)
    raise CsrlError(f"unsupported loss aggregation {settings.loss_aggregation}")


def policy_objective(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor | None,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    settings: GrpoObjectiveSettings,
) -> ObjectiveOutput:
    """Clipped surrogate loss to minimize. Tensors are [N, T] except advantages [N]."""
    if settings.kl_coefficient > 0 and reference_log_probs is None:
        raise CsrlError("kl_coefficient is positive but no reference log probs were given")
    weights = action_mask.float()
    log_ratio = (log_probs - old_log_probs) * weights
    if settings.importance_ratio is ImportanceRatio.TOKEN:
        ratio = torch.exp(log_ratio)
    elif settings.importance_ratio is ImportanceRatio.SEQUENCE:
        sequence_log_ratio = log_ratio.sum(dim=1) / weights.sum(dim=1)
        ratio = torch.exp(sequence_log_ratio).unsqueeze(1).expand_as(log_probs)
    else:
        raise CsrlError(f"unsupported importance ratio {settings.importance_ratio}")
    advantage = advantages.unsqueeze(1)
    clipped_ratio = ratio.clamp(1 - settings.clip_low, 1 + settings.clip_high)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    per_token = -surrogate
    mean_kl = 0.0
    if reference_log_probs is not None:
        kl = _kl_k3(log_probs, reference_log_probs) * weights
        mean_kl = float((kl.sum() / weights.sum()).detach())
        if settings.kl_coefficient > 0:
            per_token = per_token + settings.kl_coefficient * kl
    loss = _aggregate(per_token, action_mask, settings)
    was_clipped = (ratio != clipped_ratio) & action_mask
    clipped_fraction = float(was_clipped.float().sum() / weights.sum())
    return ObjectiveOutput(loss, mean_kl, clipped_fraction)
