"""Adapters for bodhan-ai/indic-transcribe-flex.

Two adapters live here:

1. OfficialTranscriber wraps the documented public API from the model card
   (IndicTranscribe.from_pretrained, then asr(path, lang=..., mode=...)). It is used for the
   S0 zero-shot baseline and as the ground truth in scripts/smoke_test_model.py.

2. NemoCanaryBackbone implements BackbonePort on the NeMo checkpoint shipped in the model
   repository (nemo/ folder, loaded with load_nemo_model as documented on the model card).

   STATUS: UNVERIFIED. The model code is gated and was not readable when this file was
   written. The NeMo calls below follow the public NeMo EncDecMultiTaskModel (Canary) module
   layout: preprocessor, encoder, encoder_decoder_proj, transf_decoder, log_softmax,
   tokenizer. Every one of these calls is checked by scripts/smoke_test_model.py, which must
   pass before any training run. If a call fails, fix it here and nowhere else.

The NeMo path writes native script only. Mixed mode is selected by the decoder prompt, so
the prompt ids are read from configs/model.yaml (discovered with scripts/inspect_model.py)
and the smoke test confirms that greedy decoding with that prompt reproduces the official
mixed-mode output.
"""

import copy
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from csrl.application.ports import EncoderOutput
from csrl.domain import constants
from csrl.domain.entities import Utterance
from csrl.domain.errors import ConfigError
from csrl.infrastructure.config import require
from csrl.infrastructure.logging import get_logger

_logger = get_logger(__name__)
_NEMO_SUBDIRECTORY = "nemo"
_TRAINABLE_PREFIXES = ("transf_decoder.", "log_softmax.")


@dataclass(frozen=True)
class ModelSettings:
    model_directory: Path
    language_code: str
    transcription_mode: str
    device: str
    prompt_token_ids: list[int]
    equivalence_min_exact_match: float
    equivalence_sample_count: int


def model_settings(config: dict[str, Any], require_prompt: bool) -> ModelSettings:
    prompt = [int(token) for token in require(config, "prompt_token_ids")]
    if require_prompt and not prompt:
        raise ConfigError(
            "configs/model.yaml prompt_token_ids is empty; run scripts/inspect_model.py and "
            "fill it before training or smoke testing the backbone"
        )
    return ModelSettings(
        model_directory=Path(str(require(config, "model_directory"))),
        language_code=str(require(config, "language_code")),
        transcription_mode=str(require(config, "transcription_mode")),
        device=str(require(config, "device")),
        prompt_token_ids=prompt,
        equivalence_min_exact_match=float(require(config, "equivalence_min_exact_match")),
        equivalence_sample_count=int(require(config, "equivalence_sample_count")),
    )


def _add_to_path(directory: Path) -> None:
    if not directory.is_dir():
        raise ConfigError(f"model directory not found: {directory}")
    entry = str(directory.resolve())
    if entry not in sys.path:
        sys.path.insert(0, entry)


class OfficialTranscriber:
    """Transcriber over the documented model API. Uses the model's own decoding."""

    def __init__(self, settings: ModelSettings, audio_base_directory: Path) -> None:
        _add_to_path(settings.model_directory)
        from indic_transcribe import IndicTranscribe  # type: ignore[import-not-found]

        self._model = IndicTranscribe.from_pretrained(str(settings.model_directory))
        self._settings = settings
        self._audio_base = audio_base_directory

    def transcribe(self, utterances: Sequence[Utterance]) -> list[str]:
        outputs = []
        for utterance in utterances:
            text = self._model(
                str(self._audio_base / utterance.audio_path),
                lang=self._settings.language_code,
                mode=self._settings.transcription_mode,
            )
            if not isinstance(text, str):
                raise TypeError(f"official model returned {type(text).__name__}, expected str")
            outputs.append(text)
        return outputs


class NemoTokenizer:
    """TokenizerPort over the NeMo Canary tokenizer. UNVERIFIED method names."""

    def __init__(self, tokenizer: Any, language_code: str) -> None:
        self._tokenizer = tokenizer
        self._language = language_code
        self._vocabulary = [
            tokenizer.ids_to_tokens([index])[0] for index in range(tokenizer.vocab_size)
        ]

    @property
    def vocabulary(self) -> list[str]:
        return self._vocabulary

    @property
    def eos_id(self) -> int:
        return int(self._tokenizer.eos_id)

    @property
    def pad_id(self) -> int:
        return int(self._tokenizer.pad_id)

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._tokenizer.text_to_ids(text, self._language)]

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.ids_to_text(token_ids))


class NemoCanaryBackbone:
    """BackbonePort over the NeMo checkpoint. Encoder frozen; decoder and output head train."""

    def __init__(self, model: Any, settings: ModelSettings) -> None:
        _logger.warning(constants.UNVERIFIED_BINDING_NOTICE)
        self._model = model
        self._settings = settings
        self._tokenizer = NemoTokenizer(model.tokenizer, settings.language_code)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(_TRAINABLE_PREFIXES))
        model.eval()

    @classmethod
    def load(cls, settings: ModelSettings) -> "NemoCanaryBackbone":
        _add_to_path(settings.model_directory / _NEMO_SUBDIRECTORY)
        from load_nemo import load_nemo_model  # type: ignore[import-not-found]

        model = load_nemo_model(str(settings.model_directory))
        model = model.to(torch.device(settings.device))
        return cls(model, settings)

    @property
    def tokenizer(self) -> NemoTokenizer:
        return self._tokenizer

    @property
    def hidden_size(self) -> int:
        return int(self._model.transf_decoder.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    def prompt_token_ids(self) -> list[int]:
        return list(self._settings.prompt_token_ids)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> EncoderOutput:
        features, feature_lengths = self._model.preprocessor(
            input_signal=audio, length=audio_lengths
        )
        encoded, encoded_lengths = self._model.encoder(
            audio_signal=features, length=feature_lengths
        )
        states = self._model.encoder_decoder_proj(encoded.transpose(1, 2))
        frames = states.shape[1]
        mask = torch.arange(frames, device=states.device).unsqueeze(0) < encoded_lengths.unsqueeze(1)
        return EncoderOutput(states, mask)

    def decode_hidden(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor, encoder: EncoderOutput
    ) -> torch.Tensor:
        return self._model.transf_decoder(
            input_ids=input_ids,
            decoder_mask=input_mask.to(encoder.states.dtype),
            encoder_embeddings=encoder.states,
            encoder_mask=encoder.mask.to(encoder.states.dtype),
        )

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The NeMo head returns log probabilities. They differ from logits by a per-row
        constant, so softmax, temperature scaling, and additive biases give identical
        distributions."""
        return self._model.log_softmax(hidden_states=hidden)

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        return (p for p in self._model.parameters() if p.requires_grad)

    def trainable_state_dict(self) -> Mapping[str, torch.Tensor]:
        return {
            name: tensor
            for name, tensor in self._model.state_dict().items()
            if name.startswith(_TRAINABLE_PREFIXES)
        }

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        unexpected = [name for name in state if not name.startswith(_TRAINABLE_PREFIXES)]
        if unexpected:
            raise ConfigError(f"checkpoint has non-trainable keys: {unexpected[:5]}")
        result = self._model.load_state_dict(dict(state), strict=False)
        if result.unexpected_keys:
            raise ConfigError(f"unexpected keys in checkpoint: {result.unexpected_keys[:5]}")
        missing = [k for k in result.missing_keys if k.startswith(_TRAINABLE_PREFIXES)]
        if missing:
            raise ConfigError(f"checkpoint is missing trainable keys: {missing[:5]}")

    def reference_copy(self) -> "NemoCanaryBackbone":
        """Shares the frozen encoder; deep copies only the decoder and the output head."""
        reference = copy.copy(self)
        reference._model = copy.copy(self._model)
        decoder = copy.deepcopy(self._model.transf_decoder)
        head = copy.deepcopy(self._model.log_softmax)
        for module in (decoder, head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()
        object.__setattr__(reference._model, "transf_decoder", decoder)
        object.__setattr__(reference._model, "log_softmax", head)
        return reference

    def train_mode(self, enabled: bool) -> None:
        self._model.transf_decoder.train(enabled)
        self._model.log_softmax.train(enabled)
        self._model.encoder.eval()
        self._model.preprocessor.eval()
