"""Builds TTS backends, voice pools, and filter settings from configs/synthesis.yaml."""

from typing import Any

from csrl.adapters.gemini_tts import GeminiSettings, GeminiTts
from csrl.adapters.omnivoice_tts import OmniVoiceSettings, OmniVoiceTts
from csrl.application.quality_filter import FilterSettings
from csrl.application.synthesis import SynthesisSettings, TtsPort
from csrl.data.augment import TelephonySettings
from csrl.data.sentences import SentenceLimits
from csrl.domain.entities import VoiceSpec
from csrl.domain.enums import DataSource, DeliveryStyle, RoundTripScope, TtsSystem
from csrl.domain.errors import ConfigError
from csrl.infrastructure.config import require

_SOURCE_BY_SYSTEM = {
    TtsSystem.GEMINI: DataSource.SYNTH_GEMINI,
    TtsSystem.BASE_OMNIVOICE: DataSource.SYNTH_BASE_OMNIVOICE,
    TtsSystem.INHOUSE_OMNIVOICE: DataSource.SYNTH_INHOUSE_OMNIVOICE,
}


def system_config(config: dict[str, Any], system: TtsSystem) -> dict[str, Any]:
    systems = require(config, "systems")
    return require(systems, system.value)  # type: ignore[no-any-return]


def voice_pool(config: dict[str, Any], system: TtsSystem, style: DeliveryStyle) -> list[VoiceSpec]:
    voices = []
    for entry in require(system_config(config, system), "voices"):
        voice_style = DeliveryStyle(require(entry, "style"))
        if voice_style is not style:
            continue
        voices.append(VoiceSpec(
            system=system.value,
            voice_id=str(require(entry, "voice_id")),
            style=voice_style.value,
            reference_audio=entry.get("reference_audio"),
            reference_text=entry.get("reference_text"),
        ))
    if not voices:
        raise ConfigError(f"no {style.value} voices configured for {system.value}")
    return voices


def build_tts(config: dict[str, Any], system: TtsSystem) -> TtsPort:
    section = system_config(config, system)
    if system is TtsSystem.GEMINI:
        return GeminiTts(GeminiSettings(
            model=str(require(section, "model")),
            api_key_env=str(require(section, "api_key_env")),
            style_prompts={str(k): str(v) for k, v in require(section, "style_prompts").items()},
            max_retries=int(require(section, "max_retries")),
            retry_status_codes=tuple(int(c) for c in require(section, "retry_status_codes")),
            backoff_seconds=float(require(section, "backoff_seconds")),
        ))
    plugin = require(section, "style_plugin")
    return OmniVoiceTts(OmniVoiceSettings(
        model_path=str(require(section, "model_path")),
        device=str(require(section, "device")),
        output_sample_rate=int(require(section, "output_sample_rate")),
        data_source=_SOURCE_BY_SYSTEM[system],
        style_plugin=None if plugin is None else str(plugin),
        language=str(section.get("language", "npi")),
    ))


def synthesis_settings(config: dict[str, Any]) -> SynthesisSettings:
    return SynthesisSettings(
        voices_per_sentence=int(require(config, "voices_per_sentence")),
        max_failure_fraction=float(require(config, "max_failure_fraction")),
        seed=int(require(config, "seed")),
    )


def sentence_limits(config: dict[str, Any]) -> SentenceLimits:
    limits = require(config, "sentence_limits")
    return SentenceLimits(int(require(limits, "min_words")), int(require(limits, "max_words")))


def filter_settings(config: dict[str, Any]) -> FilterSettings:
    section = require(config, "filter")
    return FilterSettings(
        max_round_trip_cer=float(require(section, "max_round_trip_cer")),
        min_duration_seconds=float(require(section, "min_duration_seconds")),
        max_duration_seconds=float(require(section, "max_duration_seconds")),
        scope=RoundTripScope(require(section, "scope")),
        min_characters_per_second=float(require(section, "min_characters_per_second")),
        max_characters_per_second=float(require(section, "max_characters_per_second")),
    )


def telephony_settings(config: dict[str, Any]) -> tuple[TelephonySettings, int, str]:
    section = require(config, "telephony")
    settings = TelephonySettings(
        snr_db_low=float(require(section, "snr_db_low")),
        snr_db_high=float(require(section, "snr_db_high")),
        seed=int(require(config, "seed")),
    )
    return (settings, int(require(section, "quantization_levels")),
            str(require(section, "output_subdirectory")))
