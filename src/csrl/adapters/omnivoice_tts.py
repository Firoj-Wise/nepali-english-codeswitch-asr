"""OmniVoice backend for the public base model and for a private fine-tuned checkpoint.

API follows https://github.com/k2-fsa/OmniVoice as read on 2026-10-19:
OmniVoice.from_pretrained(path, device_map=..., dtype=...), then
model.generate(text=..., ref_audio=..., ref_text=...) returning a list of float arrays.

Confidential model features (for example expressive style tags of a private checkpoint)
must not be written into this repository. They are applied by a style plugin: a function
`transform(text: str, style: str) -> str` in a module outside the repository, named in the
config as "module:function" and importable from PYTHONPATH. With no plugin, only the
neutral style is allowed.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from csrl.domain.entities import VoiceSpec
from csrl.domain.enums import DataSource, DeliveryStyle
from csrl.domain.errors import ConfigError, TtsError

_PLUGIN_SEPARATOR = ":"


@dataclass(frozen=True)
class OmniVoiceSettings:
    model_path: str
    device: str
    output_sample_rate: int
    data_source: DataSource
    style_plugin: str | None
    language: str = "npi"


def _load_plugin(spec: str) -> Callable[[str, str], str]:
    module_name, _, function_name = spec.partition(_PLUGIN_SEPARATOR)
    if not module_name or not function_name:
        raise ConfigError(f"style_plugin must be 'module:function', got {spec!r}")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise ConfigError(f"style_plugin {spec} is not callable")
    return function  # type: ignore[no-any-return]


class OmniVoiceTts:
    def __init__(self, settings: OmniVoiceSettings) -> None:
        import torch
        from omnivoice import OmniVoice  # type: ignore[import-not-found]

        try:
            import omnivoice.models.omnivoice as ov_mod  # type: ignore[import-not-found]

            ov_mod._resolve_instruct = lambda x, **kw: x
        except (ImportError, AttributeError):
            pass

        self._model = OmniVoice.from_pretrained(
            settings.model_path, device_map=settings.device, dtype=torch.float16
        )
        self._settings = settings
        self._plugin = _load_plugin(settings.style_plugin) if settings.style_plugin else None

    @property
    def data_source(self) -> DataSource:
        return self._settings.data_source

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        if voice.reference_audio is None:
            raise ConfigError(f"OmniVoice voice {voice.voice_id} needs reference_audio")
        prompt_text = text
        if voice.style != DeliveryStyle.NEUTRAL.value:
            if self._plugin is None:
                raise ConfigError(f"style {voice.style} requested but no style_plugin is set")
            prompt_text = self._plugin(text, voice.style)
        try:
            outputs = self._model.generate(
                text=prompt_text,
                language=self._settings.language,
                ref_audio=voice.reference_audio,
                ref_text=voice.reference_text,
            )
        except (RuntimeError, ValueError) as error:
            raise TtsError(f"OmniVoice generation failed: {error}") from error
        if not outputs:
            raise TtsError("OmniVoice returned no audio")
        waveform = np.asarray(outputs[0], dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise TtsError(f"OmniVoice returned audio of shape {waveform.shape}")
        return waveform, self._settings.output_sample_rate
