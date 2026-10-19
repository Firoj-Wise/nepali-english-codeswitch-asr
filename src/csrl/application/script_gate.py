"""Token-level script gate: predicts the script class of the next token from decoder state."""

from dataclasses import dataclass

import torch
from torch import nn

from csrl.domain import constants
from csrl.domain.enums import GateClass, Script
from csrl.metrics.script import token_script

_SCRIPT_TO_GATE = {Script.DEVANAGARI: GateClass.DEVANAGARI, Script.LATIN: GateClass.LATIN}
_CONFUSABLE_GATE_CLASSES = (GateClass.DEVANAGARI, GateClass.LATIN)


def vocabulary_gate_classes(vocabulary: list[str], special_ids: set[int]) -> torch.Tensor:
    """Gate class per token id [V]. Special tokens and non-letter pieces map to OTHER."""
    classes = []
    for token_id, piece in enumerate(vocabulary):
        if token_id in special_ids:
            classes.append(GateClass.OTHER.value)
            continue
        text = piece.replace(constants.SENTENCEPIECE_WORD_MARKER, "")
        script = token_script(text) if text else Script.OTHER
        classes.append(_SCRIPT_TO_GATE.get(script, GateClass.OTHER).value)
    return torch.tensor(classes, dtype=torch.long)


@dataclass(frozen=True)
class GateBiasSettings:
    strength: float
    confidence_threshold: float


class ScriptGate(nn.Module):
    def __init__(self, hidden_size: int, token_classes: torch.Tensor) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_size, len(GateClass))
        self.register_buffer("token_classes", token_classes, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Gate logits [B, L, len(GateClass)]."""
        return self.head(hidden)

    def auxiliary_loss(
        self, hidden: torch.Tensor, target_ids: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross entropy against the script class of each target token, over valid positions."""
        logits = self.forward(hidden)
        targets = self.token_classes[target_ids]
        losses = nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).view_as(targets)
        weights = target_mask.float()
        return (losses * weights).sum() / weights.sum()

    def logit_bias(self, hidden: torch.Tensor, settings: GateBiasSettings) -> torch.Tensor:
        """Bias [B, L, V]: subtract strength from tokens of the opposite confusable script.

        Applied only where the gate predicts a confusable class with probability at least
        the confidence threshold. Elsewhere the bias is zero.
        """
        probabilities = self.forward(hidden).softmax(dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        vocabulary_size = self.token_classes.shape[0]
        bias_rows = torch.zeros(len(GateClass), vocabulary_size, device=hidden.device)
        for gate_class in _CONFUSABLE_GATE_CLASSES:
            opposite = (self.token_classes != gate_class.value) & torch.isin(
                self.token_classes,
                torch.tensor([c.value for c in _CONFUSABLE_GATE_CLASSES], device=hidden.device),
            )
            bias_rows[gate_class.value] = -settings.strength * opposite.float()
        active = (confidence >= settings.confidence_threshold).float().unsqueeze(-1)
        return active * bias_rows[predicted]
