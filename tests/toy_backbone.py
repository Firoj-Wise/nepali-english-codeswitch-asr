"""A tiny causal encoder-decoder implementing BackbonePort, for CPU tests only."""

import copy
from collections.abc import Iterator, Mapping

import torch
from torch import nn

from csrl.application.ports import EncoderOutput

VOCABULARY = ["<pad>", "<eos>", "<ne>", "▁मेरो", "▁phone", "▁फोन", "▁राम्रो"]
PAD_ID, EOS_ID, PROMPT_ID = 0, 1, 2
FRAME = 160
HIDDEN = 16


class ToyTokenizer:
    @property
    def vocabulary(self) -> list[str]:
        return VOCABULARY

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def encode(self, text: str) -> list[int]:
        lookup = {piece.lstrip("▁"): index for index, piece in enumerate(VOCABULARY)}
        return [lookup[word] for word in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(VOCABULARY[i].lstrip("▁") for i in token_ids)


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(FRAME, HIDDEN)
        self.embedding = nn.Embedding(len(VOCABULARY), HIDDEN)
        self.mix = nn.Linear(2 * HIDDEN, HIDDEN)
        self.output = nn.Linear(HIDDEN, len(VOCABULARY))
        self._tokenizer = ToyTokenizer()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def tokenizer(self) -> ToyTokenizer:
        return self._tokenizer

    @property
    def hidden_size(self) -> int:
        return HIDDEN

    @property
    def device(self) -> torch.device:
        return self.output.weight.device

    def prompt_token_ids(self) -> list[int]:
        return [PROMPT_ID]

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> EncoderOutput:
        frames = audio.shape[1] // FRAME
        chunks = audio[:, : frames * FRAME].view(audio.shape[0], frames, FRAME)
        mask = torch.arange(frames).unsqueeze(0) < (audio_lengths // FRAME).unsqueeze(1)
        return EncoderOutput(self.encoder(chunks), mask)

    def decode_hidden(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor, encoder: EncoderOutput
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids) * input_mask.unsqueeze(-1).float()
        counts = torch.arange(1, input_ids.shape[1] + 1, device=input_ids.device).view(1, -1, 1)
        causal_mean = embedded.cumsum(dim=1) / counts
        weights = encoder.mask.unsqueeze(-1).float()
        context = (encoder.states * weights).sum(1) / weights.sum(1).clamp_min(1)
        context = context.unsqueeze(1).expand_as(causal_mean)
        return torch.tanh(self.mix(torch.cat([causal_mean, context], dim=-1)))

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_state_dict(self) -> Mapping[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(dict(state), strict=False)

    def reference_copy(self) -> "ToyBackbone":
        reference = copy.deepcopy(self)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    def train_mode(self, enabled: bool) -> None:
        self.train(enabled)
        self.encoder.eval()
