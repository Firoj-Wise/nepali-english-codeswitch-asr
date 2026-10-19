"""Decoding, log probabilities, and supervised loss on top of a BackbonePort.

The same logit transform (temperature and optional gate bias) is used for sampling and for
log probabilities, so the GRPO update is computed under the distribution that was sampled.
Generation runs without a key-value cache: each step re-runs the decoder on the full prefix.
This is correct and simple; it is slower than cached decoding.
"""

from dataclasses import dataclass

import torch

from csrl.application.ports import BackbonePort, EncoderOutput
from csrl.application.script_gate import GateBiasSettings, ScriptGate
from csrl.domain.errors import CsrlError

_GREEDY_TEMPERATURE = 1.0


@dataclass(frozen=True)
class Generation:
    content_ids: torch.Tensor  # [N, T], padded with pad_id after eos
    action_mask: torch.Tensor  # [N, T], True for generated tokens including eos
    texts: list[str]


@dataclass(frozen=True)
class SupervisedLoss:
    token_loss: torch.Tensor
    gate_loss: torch.Tensor | None


class Policy:
    def __init__(
        self,
        backbone: BackbonePort,
        gate: ScriptGate | None,
        gate_bias: GateBiasSettings | None,
        max_new_tokens: int,
    ) -> None:
        if gate_bias is not None and gate is None:
            raise CsrlError("gate bias settings given without a gate")
        self.backbone = backbone
        self.gate = gate
        self.gate_bias = gate_bias
        self.max_new_tokens = max_new_tokens

    def _logits(self, hidden: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = self.backbone.project_logits(hidden)
        if self.gate is not None and self.gate_bias is not None:
            logits = logits + self.gate.logit_bias(hidden, self.gate_bias)
        return logits / temperature

    def _prompt(self, rows: int) -> torch.Tensor:
        prompt = torch.tensor(self.backbone.prompt_token_ids(), device=self.backbone.device)
        return prompt.unsqueeze(0).expand(rows, -1)

    @torch.no_grad()
    def generate(self, encoder: EncoderOutput, temperature: float, greedy: bool) -> Generation:
        tokenizer = self.backbone.tokenizer
        rows = encoder.states.shape[0]
        sequence = self._prompt(rows).clone()
        finished = torch.zeros(rows, dtype=torch.bool, device=sequence.device)
        generated: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for _ in range(self.max_new_tokens):
            input_mask = torch.ones_like(sequence, dtype=torch.bool)
            hidden = self.backbone.decode_hidden(sequence, input_mask, encoder)[:, -1:, :]
            effective_temperature = _GREEDY_TEMPERATURE if greedy else temperature
            logits = self._logits(hidden, effective_temperature)[:, -1, :]
            if greedy:
                next_ids = logits.argmax(dim=-1)
            else:
                next_ids = torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, tokenizer.pad_id), next_ids)
            masks.append(~finished)
            generated.append(next_ids)
            finished = finished | (next_ids == tokenizer.eos_id)
            sequence = torch.cat([sequence, next_ids.unsqueeze(-1)], dim=-1)
            if bool(finished.all()):
                break
        content_ids = torch.stack(generated, dim=1)
        action_mask = torch.stack(masks, dim=1)
        texts = [self._text(row, mask) for row, mask in zip(content_ids, action_mask)]
        return Generation(content_ids, action_mask, texts)

    def _text(self, row: torch.Tensor, mask: torch.Tensor) -> str:
        tokenizer = self.backbone.tokenizer
        ids = [int(t) for t, valid in zip(row.tolist(), mask.tolist()) if valid]
        if ids and ids[-1] == tokenizer.eos_id:
            ids = ids[:-1]
        return tokenizer.decode(ids)

    def token_log_probs(
        self,
        encoder: EncoderOutput,
        content_ids: torch.Tensor,
        action_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Log probability of each content token [N, T] under the policy. Gradient flows."""
        prompt = self._prompt(content_ids.shape[0])
        prompt_length = prompt.shape[1]
        input_ids = torch.cat([prompt, content_ids], dim=1)[:, :-1]
        input_mask = torch.cat(
            [torch.ones_like(prompt, dtype=torch.bool), action_mask], dim=1
        )[:, :-1]
        hidden = self.backbone.decode_hidden(input_ids, input_mask, encoder)
        hidden = hidden[:, prompt_length - 1 :, :]
        log_probs = self._logits(hidden, temperature).log_softmax(dim=-1)
        gathered = log_probs.gather(-1, content_ids.unsqueeze(-1)).squeeze(-1)
        return gathered * action_mask.float()

    def supervised_loss(self, encoder: EncoderOutput, references: list[str]) -> SupervisedLoss:
        """Token cross entropy on reference transcripts plus the gate auxiliary loss if present."""
        tokenizer = self.backbone.tokenizer
        encoded = [tokenizer.encode(text) + [tokenizer.eos_id] for text in references]
        length = max(len(ids) for ids in encoded)
        device = self.backbone.device
        targets = torch.full((len(encoded), length), tokenizer.pad_id, device=device)
        target_mask = torch.zeros((len(encoded), length), dtype=torch.bool, device=device)
        for row, ids in enumerate(encoded):
            targets[row, : len(ids)] = torch.tensor(ids, device=device)
            target_mask[row, : len(ids)] = True
        prompt = self._prompt(len(encoded))
        prompt_length = prompt.shape[1]
        input_ids = torch.cat([prompt, targets], dim=1)[:, :-1]
        input_mask = torch.cat([torch.ones_like(prompt, dtype=torch.bool), target_mask], dim=1)[
            :, :-1
        ]
        hidden = self.backbone.decode_hidden(input_ids, input_mask, encoder)
        hidden = hidden[:, prompt_length - 1 :, :]
        log_probs = self.backbone.project_logits(hidden).log_softmax(dim=-1)
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        weights = target_mask.float()
        token_loss = (token_nll * weights).sum() / weights.sum()
        gate_loss = None
        if self.gate is not None:
            gate_loss = self.gate.auxiliary_loss(hidden, targets, target_mask)
        return SupervisedLoss(token_loss, gate_loss)
