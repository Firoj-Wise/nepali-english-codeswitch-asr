"""Checkpoint save and load for trainable parameters and the script gate."""

from pathlib import Path

import torch

from csrl.application.ports import BackbonePort
from csrl.application.script_gate import ScriptGate
from csrl.domain import constants
from csrl.domain.enums import CheckpointKey
from csrl.domain.errors import CsrlError


def save_checkpoint(
    path: Path, backbone: BackbonePort, gate: ScriptGate | None, step: int, dev_mer: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        CheckpointKey.BACKBONE.value: dict(backbone.trainable_state_dict()),
        CheckpointKey.GATE.value: gate.state_dict() if gate is not None else None,
        CheckpointKey.STEP.value: step,
        CheckpointKey.DEV_MER.value: dev_mer,
        CheckpointKey.METRIC_VERSION.value: constants.METRIC_VERSION,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, backbone: BackbonePort, gate: ScriptGate | None) -> int:
    """Load weights in place and return the training step stored in the checkpoint."""
    if not path.is_file():
        raise CsrlError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=backbone.device, weights_only=True)
    if payload[CheckpointKey.METRIC_VERSION.value] != constants.METRIC_VERSION:
        raise CsrlError(f"checkpoint {path} was selected under a different metric version")
    backbone.load_trainable_state_dict(payload[CheckpointKey.BACKBONE.value])
    gate_state = payload[CheckpointKey.GATE.value]
    if (gate is None) != (gate_state is None):
        raise CsrlError(f"checkpoint {path}: gate presence does not match the configuration")
    if gate is not None:
        gate.load_state_dict(gate_state)
    return int(payload[CheckpointKey.STEP.value])
