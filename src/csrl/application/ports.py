"""Ports that a model backbone must implement. The training code depends only on these.

Shapes use B for batch, L for decoder sequence length, S for encoder frames, H for hidden
size, and V for vocabulary size.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class EncoderOutput:
    states: torch.Tensor  # [B, S, H_enc]
    mask: torch.Tensor  # [B, S], True for valid frames

    def repeat_each(self, times: int) -> "EncoderOutput":
        """Repeat every row `times` times consecutively: rows a, b become a, a, b, b."""
        return EncoderOutput(
            self.states.repeat_interleave(times, dim=0),
            self.mask.repeat_interleave(times, dim=0),
        )


class TokenizerPort(Protocol):
    @property
    def vocabulary(self) -> list[str]:
        """Token string for every id, index aligned with the logits."""
        ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    def encode(self, text: str) -> list[int]:
        """Token ids for transcript text, without prompt or end-of-sequence tokens."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Text for content token ids. Callers strip prompt, eos, and padding first."""
        ...


class BackbonePort(Protocol):
    """An encoder-decoder ASR model with a frozen encoder and a trainable decoder."""

    @property
    def tokenizer(self) -> TokenizerPort: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def prompt_token_ids(self) -> list[int]:
        """Decoder prefix that selects language and transcription mode."""
        ...

    def encode(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> EncoderOutput:
        """Run the frozen encoder on 16 kHz waveforms [B, T_audio]. No gradient."""
        ...

    def decode_hidden(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor, encoder: EncoderOutput
    ) -> torch.Tensor:
        """Causal decoder over input_ids [B, L]. Returns final hidden states [B, L, H]."""
        ...

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Unnormalized vocabulary logits [B, L, V] from hidden states."""
        ...

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def trainable_state_dict(self) -> Mapping[str, torch.Tensor]: ...

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None: ...

    def reference_copy(self) -> "BackbonePort":
        """A frozen copy of the decoder for the KL reference. May share the encoder."""
        ...

    def train_mode(self, enabled: bool) -> None:
        """Set train or eval mode on the trainable parts. The encoder stays in eval mode."""
        ...
