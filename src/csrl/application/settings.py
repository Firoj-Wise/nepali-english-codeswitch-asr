"""Typed run settings built from YAML mappings. Every key is required; nothing is defaulted."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from csrl.application.grpo import GrpoObjectiveSettings
from csrl.application.script_gate import GateBiasSettings
from csrl.domain.enums import AdvantageNormalization, ImportanceRatio, LossAggregation
from csrl.domain.errors import ConfigError
from csrl.infrastructure.config import require

E = TypeVar("E", bound=Enum)


def _enum(enum_type: type[E], value: Any, key: str) -> E:
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = [member.value for member in enum_type]
        raise ConfigError(f"{key}: {value!r} is not one of {allowed}") from error


def _positive(value: float, key: str) -> float:
    if value <= 0:
        raise ConfigError(f"{key} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class CommonSettings:
    seed: int
    manifest_base_directory: str
    train_manifest: str
    dev_manifest: str
    batch_size: int
    learning_rate: float
    max_steps: int
    eval_every_steps: int
    max_grad_norm: float
    max_new_tokens: int
    gate_enabled: bool
    gate_bias: GateBiasSettings | None


@dataclass(frozen=True)
class SftSettings:
    common: CommonSettings
    gate_aux_loss_weight: float


@dataclass(frozen=True)
class GrpoSettings:
    common: CommonSettings
    init_checkpoint: str
    group_size: int
    sampling_temperature: float
    update_epochs: int
    dynamic_sampling: bool
    max_empty_fraction: float
    lambda_scr: float
    lambda_hal: float
    objective: GrpoObjectiveSettings
    log_samples_every: int


def common_settings(config: dict[str, Any]) -> CommonSettings:
    gate = require(config, "script_gate")
    gate_enabled = bool(require(gate, "enabled"))
    bias = require(gate, "inference_bias")
    gate_bias = None
    if bool(require(bias, "enabled")):
        if not gate_enabled:
            raise ConfigError("script_gate.inference_bias.enabled requires script_gate.enabled")
        gate_bias = GateBiasSettings(
            strength=_positive(float(require(bias, "strength")), "inference_bias.strength"),
            confidence_threshold=float(require(bias, "confidence_threshold")),
        )
    return CommonSettings(
        seed=int(require(config, "seed")),
        manifest_base_directory=str(require(config, "manifest_base_directory")),
        train_manifest=str(require(config, "train_manifest")),
        dev_manifest=str(require(config, "dev_manifest")),
        batch_size=int(_positive(int(require(config, "batch_size")), "batch_size")),
        learning_rate=_positive(float(require(config, "learning_rate")), "learning_rate"),
        max_steps=int(_positive(int(require(config, "max_steps")), "max_steps")),
        eval_every_steps=int(_positive(int(require(config, "eval_every_steps")), "eval_every")),
        max_grad_norm=_positive(float(require(config, "max_grad_norm")), "max_grad_norm"),
        max_new_tokens=int(_positive(int(require(config, "max_new_tokens")), "max_new_tokens")),
        gate_enabled=gate_enabled,
        gate_bias=gate_bias,
    )


def sft_settings(config: dict[str, Any]) -> SftSettings:
    gate = require(config, "script_gate")
    return SftSettings(common_settings(config), float(require(gate, "aux_loss_weight")))


def grpo_settings(config: dict[str, Any]) -> GrpoSettings:
    objective = require(config, "objective")
    reward = require(config, "reward")
    settings = GrpoObjectiveSettings(
        advantage_normalization=_enum(
            AdvantageNormalization, require(objective, "advantage_normalization"),
            "advantage_normalization"),
        loss_aggregation=_enum(
            LossAggregation, require(objective, "loss_aggregation"), "loss_aggregation"),
        importance_ratio=_enum(
            ImportanceRatio, require(objective, "importance_ratio"), "importance_ratio"),
        clip_low=_positive(float(require(objective, "clip_low")), "clip_low"),
        clip_high=_positive(float(require(objective, "clip_high")), "clip_high"),
        kl_coefficient=float(require(objective, "kl_coefficient")),
        constant_normalizer=int(require(objective, "constant_normalizer")),
    )
    if settings.kl_coefficient < 0:
        raise ConfigError("kl_coefficient must be zero or positive")
    if (settings.importance_ratio is ImportanceRatio.SEQUENCE
            and settings.loss_aggregation is not LossAggregation.SEQUENCE_MEAN):
        raise ConfigError("GSPO (importance_ratio: sequence) requires loss_aggregation: "
                          "sequence_mean")
    group_size = int(require(config, "group_size"))
    if group_size < 2:
        raise ConfigError("group_size must be at least 2 for group-relative advantages")
    return GrpoSettings(
        common=common_settings(config),
        init_checkpoint=str(require(config, "init_checkpoint")),
        group_size=group_size,
        sampling_temperature=_positive(
            float(require(config, "sampling_temperature")), "sampling_temperature"),
        update_epochs=int(_positive(int(require(config, "update_epochs")), "update_epochs")),
        dynamic_sampling=bool(require(config, "dynamic_sampling")),
        max_empty_fraction=float(require(config, "max_empty_fraction")),
        lambda_scr=float(require(reward, "lambda_scr")),
        lambda_hal=float(require(reward, "lambda_hal")),
        objective=settings,
        log_samples_every=int(require(config, "log_samples_every")),
    )
