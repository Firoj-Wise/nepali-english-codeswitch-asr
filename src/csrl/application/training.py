"""SFT and GRPO training loops over a BackbonePort, plus dev evaluation with the policy."""

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from csrl.application.checkpoint import save_checkpoint
from csrl.application.evaluate import UtteranceScores, score_pairs, summarize
from csrl.application.grpo import (
    group_advantages,
    nonconstant_group_mask,
    policy_objective,
)
from csrl.application.policy import Policy
from csrl.application.settings import CommonSettings, GrpoSettings, SftSettings
from csrl.data.normalize import normalize_text
from csrl.domain import constants
from csrl.domain.entities import Utterance
from csrl.domain.enums import HypothesisField, LogKey, MetricName, Split
from csrl.domain.errors import EmptyReferenceError, RunHaltedError
from csrl.infrastructure.audio import collate_audio
from csrl.infrastructure.logging import get_logger
from csrl.metrics.reward import RewardWeights, compute_reward

_logger = get_logger(__name__)
_SAMPLE_PREVIEW_COUNT = 2


@dataclass(frozen=True)
class RunPaths:
    run_directory: Path
    audio_base_directory: Path

    @property
    def best_checkpoint(self) -> Path:
        return self.run_directory / "checkpoints" / constants.BEST_CHECKPOINT_NAME

    @property
    def last_checkpoint(self) -> Path:
        return self.run_directory / "checkpoints" / constants.LAST_CHECKPOINT_NAME


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    _logger.info("seed set to %d", seed)


def batches(utterances: Sequence[Utterance], size: int, seed: int) -> Iterator[list[Utterance]]:
    """Endless shuffled batches. Each pass over the data uses a new permutation."""
    generator = random.Random(seed)
    order = list(utterances)
    while True:
        generator.shuffle(order)
        for start in range(0, len(order) - size + 1, size):
            yield order[start : start + size]


def validate_references(utterances: Sequence[Utterance]) -> None:
    for utterance in utterances:
        if not normalize_text(utterance.text):
            raise EmptyReferenceError(f"{utterance.utterance_id}: empty after normalization")


def evaluate_policy(
    policy: Policy,
    utterances: Sequence[Utterance],
    paths: RunPaths,
    batch_size: int,
) -> tuple[dict[str, float | int], list[UtteranceScores]]:
    policy.backbone.train_mode(False)
    hypotheses: list[str] = []
    for start in range(0, len(utterances), batch_size):
        chunk = utterances[start : start + batch_size]
        audio, lengths = collate_audio(chunk, paths.audio_base_directory, policy.backbone.device)
        encoder = policy.backbone.encode(audio, lengths)
        hypotheses.extend(policy.generate(encoder, temperature=1.0, greedy=True).texts)
    scores = score_pairs(utterances, hypotheses)
    return summarize(scores), scores


def write_evaluation(
    directory: Path,
    split: Split,
    summary: dict[str, float | int],
    scores: Sequence[UtteranceScores],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / constants.EVAL_FILE_TEMPLATE.format(split=split.value)).write_text(
        json.dumps(summary, indent=2), encoding=constants.TEXT_ENCODING
    )
    lines = [
        json.dumps(
            {
                HypothesisField.UTTERANCE_ID.value: s.utterance_id,
                HypothesisField.REFERENCE.value: s.reference,
                HypothesisField.HYPOTHESIS.value: s.hypothesis,
            },
            ensure_ascii=False,
        )
        for s in scores
    ]
    (directory / constants.HYPOTHESES_FILE_TEMPLATE.format(split=split.value)).write_text(
        "\n".join(lines) + "\n", encoding=constants.TEXT_ENCODING
    )


def _optimizer(policy: Policy, learning_rate: float) -> torch.optim.Optimizer:
    parameters = list(policy.backbone.trainable_parameters())
    if policy.gate is not None:
        parameters.extend(policy.gate.parameters())
    return torch.optim.AdamW(parameters, lr=learning_rate)


def _clip_and_step(
    policy: Policy, optimizer: torch.optim.Optimizer, common: CommonSettings
) -> None:
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(parameters, common.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _set_train(policy: Policy, enabled: bool) -> None:
    policy.backbone.train_mode(enabled)
    if policy.gate is not None:
        policy.gate.train(enabled)


def _maybe_evaluate(
    step: int,
    common: CommonSettings,
    policy: Policy,
    dev: Sequence[Utterance],
    paths: RunPaths,
    best_mer: float,
    train_mode_after: bool,
) -> float:
    if step % common.eval_every_steps != 0 and step != common.max_steps:
        return best_mer
    _set_train(policy, False)
    summary, scores = evaluate_policy(policy, dev, paths, common.batch_size)
    dev_mer = float(summary[MetricName.MER.value])
    save_checkpoint(paths.last_checkpoint, policy.backbone, policy.gate, step, dev_mer)
    if dev_mer < best_mer:
        save_checkpoint(paths.best_checkpoint, policy.backbone, policy.gate, step, dev_mer)
        write_evaluation(paths.run_directory, Split.DEV, summary, scores)
        _logger.info("step %d: new best dev MER %.4f", step, dev_mer)
        best_mer = dev_mer
    _set_train(policy, train_mode_after)
    return best_mer


def train_sft(
    policy: Policy,
    settings: SftSettings,
    train: Sequence[Utterance],
    dev: Sequence[Utterance],
    paths: RunPaths,
) -> float:
    """Returns the best dev MER. The best checkpoint is written under the run directory."""
    common = settings.common
    set_seed(common.seed)
    validate_references(train)
    validate_references(dev)
    optimizer = _optimizer(policy, common.learning_rate)
    _set_train(policy, True)
    best_mer = float("inf")
    stream = batches(train, common.batch_size, common.seed)
    for step in range(1, common.max_steps + 1):
        batch = next(stream)
        audio, lengths = collate_audio(batch, paths.audio_base_directory, policy.backbone.device)
        encoder = policy.backbone.encode(audio, lengths)
        result = policy.supervised_loss(encoder, [normalize_text(u.text) for u in batch])
        loss = result.token_loss
        if result.gate_loss is not None:
            loss = loss + settings.gate_aux_loss_weight * result.gate_loss
        loss.backward()
        _clip_and_step(policy, optimizer, common)
        _logger.info(
            "%s=%d %s=%.4f %s=%s",
            LogKey.STEP.value, step, LogKey.CE_LOSS.value, float(result.token_loss.detach()),
            LogKey.GATE_LOSS.value,
            "disabled" if result.gate_loss is None else f"{float(result.gate_loss.detach()):.4f}",
        )
        best_mer = _maybe_evaluate(step, common, policy, dev, paths, best_mer, True)
    return best_mer


def train_grpo(
    policy: Policy,
    reference: Policy | None,
    settings: GrpoSettings,
    train: Sequence[Utterance],
    dev: Sequence[Utterance],
    paths: RunPaths,
) -> float:
    """GRPO-family training. The caller loads init_checkpoint into policy and reference first.

    Dropout stays disabled (eval mode) for sampling and for updates, so that with one update
    epoch the importance ratio is exactly one. Gradients flow in eval mode.
    """
    common = settings.common
    objective = settings.objective
    if objective.kl_coefficient > 0 and reference is None:
        raise RunHaltedError("kl_coefficient is positive but no reference policy was built")
    set_seed(common.seed)
    validate_references(train)
    validate_references(dev)
    weights = RewardWeights(settings.lambda_scr, settings.lambda_hal)
    optimizer = _optimizer(policy, common.learning_rate)
    best_mer = float("inf")
    stream = batches(train, common.batch_size, common.seed)
    for step in range(1, common.max_steps + 1):
        batch = next(stream)
        audio, lengths = collate_audio(batch, paths.audio_base_directory, policy.backbone.device)
        encoder = policy.backbone.encode(audio, lengths).repeat_each(settings.group_size)
        references = [normalize_text(u.text) for u in batch for _ in range(settings.group_size)]

        _set_train(policy, False)
        generation = policy.generate(encoder, settings.sampling_temperature, greedy=False)
        hypotheses = [normalize_text(text) for text in generation.texts]
        breakdowns = [compute_reward(r, h, weights) for r, h in zip(references, hypotheses)]
        rewards = torch.tensor([b.total for b in breakdowns], device=encoder.states.device)

        empty_fraction = sum(not h for h in hypotheses) / len(hypotheses)
        if empty_fraction > settings.max_empty_fraction:
            raise RunHaltedError(
                f"step {step}: empty hypothesis fraction {empty_fraction:.3f} exceeds "
                f"{settings.max_empty_fraction}"
            )

        keep = torch.ones_like(rewards, dtype=torch.bool)
        varied = nonconstant_group_mask(rewards, settings.group_size)
        zero_variance_groups = int((~varied).sum()) // settings.group_size
        if settings.dynamic_sampling:
            keep = varied
        advantages = group_advantages(
            rewards, settings.group_size, objective.advantage_normalization
        )

        if bool(keep.any()):
            with torch.no_grad():
                old_log_probs = policy.token_log_probs(
                    encoder, generation.content_ids, generation.action_mask,
                    settings.sampling_temperature,
                )
                reference_log_probs = None
                if reference is not None:
                    reference_log_probs = reference.token_log_probs(
                        encoder, generation.content_ids, generation.action_mask,
                        settings.sampling_temperature,
                    )
            for _ in range(settings.update_epochs):
                rows = keep.nonzero().squeeze(-1)
                kept_encoder = type(encoder)(encoder.states[rows], encoder.mask[rows])
                log_probs = policy.token_log_probs(
                    kept_encoder, generation.content_ids[rows], generation.action_mask[rows],
                    settings.sampling_temperature,
                )
                output = policy_objective(
                    log_probs,
                    old_log_probs[rows],
                    None if reference_log_probs is None else reference_log_probs[rows],
                    advantages[rows],
                    generation.action_mask[rows],
                    objective,
                )
                output.loss.backward()
                _clip_and_step(policy, optimizer, common)
            _logger.info(
                "%s=%d %s=%.4f %s=%.4f %s=%.4f %s=%.4f %s=%.4f %s=%.4f %s=%.3f %s=%d clipped=%.3f",
                LogKey.STEP.value, step,
                LogKey.LOSS.value, float(output.loss.detach()),
                LogKey.REWARD_TOTAL.value, float(rewards.mean()),
                LogKey.REWARD_MER.value, sum(b.mer_term for b in breakdowns) / len(breakdowns),
                LogKey.REWARD_SCR.value, sum(b.scr_term for b in breakdowns) / len(breakdowns),
                LogKey.REWARD_HALLUCINATION.value,
                sum(b.hallucination_term for b in breakdowns) / len(breakdowns),
                LogKey.KL.value, output.mean_kl,
                LogKey.EMPTY_FRACTION.value, empty_fraction,
                LogKey.ZERO_VARIANCE_GROUPS.value, zero_variance_groups,
                output.clipped_fraction,
            )
        else:
            _logger.warning(
                "step %d: all %d groups had constant reward; no update",
                step, zero_variance_groups,
            )
        if step % settings.log_samples_every == 0:
            for reference_text, hypothesis in list(zip(references, hypotheses))[
                :: settings.group_size
            ][:_SAMPLE_PREVIEW_COUNT]:
                _logger.info("sample ref=%r hyp=%r", reference_text, hypothesis)
        best_mer = _maybe_evaluate(step, common, policy, dev, paths, best_mer, False)
    return best_mer
