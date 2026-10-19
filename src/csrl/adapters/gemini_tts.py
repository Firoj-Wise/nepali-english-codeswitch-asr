"""Gemini TTS backend (google-genai SDK, Interactions API).

Request shape follows https://ai.google.dev/gemini-api/docs/speech-generation as read on
2026-10-19: client.interactions.create(model=..., input=[user_input with a text block],
response_format={"type": "audio"}, generation_config={"speech_config": [{"voice": name}]}).
Sustained delivery style is set through a speech_metadata annotation on the text block.
The response audio is base64 WAV at interaction.output_audio.data.

The API key is read from the environment variable named in the config. It is never stored
in a config file or in this repository.
"""

import base64
import io
import os
import time
from dataclasses import dataclass

import numpy as np
import soundfile

from csrl.domain.entities import VoiceSpec
from csrl.domain.enums import DataSource, DeliveryStyle
from csrl.domain.errors import ConfigError, TtsError
from csrl.infrastructure.logging import get_logger

_logger = get_logger(__name__)
_FLOAT_DTYPE = "float32"
# Request schema keys and values of the Interactions API.
_TYPE = "type"
_TEXT = "text"
_USER_INPUT = "user_input"
_CONTENT = "content"
_ANNOTATIONS = "annotations"
_SPEECH_METADATA = "speech_metadata"
_STYLE = "style"
_AUDIO = "audio"
_SPEECH_CONFIG = "speech_config"
_VOICE = "voice"
_CODE_ATTRIBUTE = "code"


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    api_key_env: str
    style_prompts: dict[str, str]
    max_retries: int
    retry_status_codes: tuple[int, ...]
    backoff_seconds: float


class GeminiTts:
    def __init__(self, settings: GeminiSettings) -> None:
        from google import genai  # type: ignore[import-not-found]
        from google.genai import errors  # type: ignore[import-not-found]

        api_key = os.environ.get(settings.api_key_env)
        if not api_key:
            raise ConfigError(f"environment variable {settings.api_key_env} is not set")
        self._client = genai.Client(api_key=api_key)
        self._api_error = errors.APIError
        self._settings = settings

    @property
    def data_source(self) -> DataSource:
        return DataSource.SYNTH_GEMINI

    def _request_input(self, text: str, voice: VoiceSpec) -> list[dict[str, object]]:
        block: dict[str, object] = {_TYPE: _TEXT, _TEXT: text}
        if voice.style != DeliveryStyle.NEUTRAL.value:
            if voice.style not in self._settings.style_prompts:
                raise ConfigError(f"no style prompt configured for style {voice.style}")
            block[_ANNOTATIONS] = [
                {_TYPE: _SPEECH_METADATA, _STYLE: self._settings.style_prompts[voice.style]}
            ]
        return [{_TYPE: _USER_INPUT, _CONTENT: [block]}]

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        attempt = 0
        while True:
            try:
                interaction = self._client.interactions.create(
                    model=self._settings.model,
                    input=self._request_input(text, voice),
                    response_format={_TYPE: _AUDIO},
                    generation_config={_SPEECH_CONFIG: [{_VOICE: voice.voice_id}]},
                )
                break
            except self._api_error as error:
                code = getattr(error, _CODE_ATTRIBUTE, None)
                if code in self._settings.retry_status_codes and attempt < self._settings.max_retries:
                    delay = self._settings.backoff_seconds * (2**attempt)
                    _logger.warning("Gemini status %s, retry %d in %.1f s", code, attempt + 1, delay)
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise TtsError(f"Gemini API error {code}: {error}") from error
        if interaction.output_audio is None:
            raise TtsError("Gemini response contained no audio")
        payload = base64.b64decode(interaction.output_audio.data)
        samples, sample_rate = soundfile.read(io.BytesIO(payload), dtype=_FLOAT_DTYPE)
        if samples.ndim != 1:
            raise TtsError(f"expected mono audio, got shape {samples.shape}")
        return samples, int(sample_rate)
