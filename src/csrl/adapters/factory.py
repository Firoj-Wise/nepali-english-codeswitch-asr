"""Composition root: builds policies from configs. Only scripts call this module."""

from dataclasses import dataclass

from csrl.adapters.indic_transcribe import ModelSettings, NemoCanaryBackbone
from csrl.application.policy import Policy
from csrl.application.script_gate import ScriptGate, vocabulary_gate_classes
from csrl.application.settings import CommonSettings


@dataclass(frozen=True)
class BuiltPolicy:
    policy: Policy
    special_ids: set[int]


def special_token_ids(backbone: NemoCanaryBackbone) -> set[int]:
    """Prompt, eos, and pad ids are never content and map to the OTHER gate class."""
    tokenizer = backbone.tokenizer
    return set(backbone.prompt_token_ids()) | {tokenizer.eos_id, tokenizer.pad_id}


def build_policy(model: ModelSettings, common: CommonSettings) -> Policy:
    backbone = NemoCanaryBackbone.load(model)
    gate = None
    if common.gate_enabled:
        classes = vocabulary_gate_classes(
            backbone.tokenizer.vocabulary, special_token_ids(backbone)
        ).to(backbone.device)
        gate = ScriptGate(backbone.hidden_size, classes).to(backbone.device)
    return Policy(backbone, gate, common.gate_bias, common.max_new_tokens)


def build_reference(policy: Policy) -> Policy:
    """Frozen copy of the current policy (call after loading the init checkpoint)."""
    gate = None
    if policy.gate is not None:
        gate = ScriptGate(policy.backbone.hidden_size, policy.gate.token_classes)
        gate.load_state_dict(policy.gate.state_dict())
        gate = gate.to(policy.backbone.device).eval()
        for parameter in gate.parameters():
            parameter.requires_grad_(False)
    return Policy(policy.backbone.reference_copy(), gate, policy.gate_bias, policy.max_new_tokens)
