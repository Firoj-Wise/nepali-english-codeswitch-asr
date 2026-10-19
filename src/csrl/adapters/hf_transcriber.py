"""Independent ASR for the round-trip filter, through a Hugging Face transformers pipeline."""

from collections.abc import Sequence
from pathlib import Path

from csrl.domain import constants
from csrl.domain.entities import Utterance
from csrl.domain.errors import ConfigError
from csrl.infrastructure.audio import load_waveform

_TASK = "automatic-speech-recognition"
_TEXT_KEY = "text"
_STUDY_MODEL_MARKER = "indic-transcribe"
_RAW_KEY = "raw"
_RATE_KEY = "sampling_rate"


class HuggingFaceTranscriber:
    def __init__(self, model_id: str, device: str, batch_size: int, audio_base: Path) -> None:
        if _STUDY_MODEL_MARKER in model_id:
            raise ConfigError("the filter ASR must be independent of the model under study")
        from transformers import pipeline  # type: ignore[import-not-found]

        self._pipeline = pipeline(_TASK, model=model_id, device=device)
        self._batch_size = batch_size
        self._audio_base = audio_base

    def transcribe(self, utterances: Sequence[Utterance]) -> list[str]:
        inputs = [
            {_RAW_KEY: load_waveform(self._audio_base / u.audio_path),
             _RATE_KEY: constants.SAMPLE_RATE_HZ}
            for u in utterances
        ]
        outputs = self._pipeline(inputs, batch_size=self._batch_size)
        return [str(item[_TEXT_KEY]) for item in outputs]
