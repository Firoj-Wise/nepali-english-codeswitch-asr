"""Synthetic speech generation through a TtsPort, with resumable manifests.

Each sentence is rendered by voices_per_sentence voices chosen deterministically from the
voice pool. A TTS failure for one request is written to failures.jsonl with its reason and
the run continues; the run raises if failures exceed max_failure_fraction. Nothing is
skipped silently.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from csrl.data.augment import resample
from csrl.data.manifest import load_manifest, serialize
from csrl.domain import constants
from csrl.domain.entities import Rejection, SentenceRecord, Utterance, VoiceSpec
from csrl.domain.enums import DataSource, RejectReason
from csrl.domain.errors import ConfigError, SynthesisBudgetError, TtsError
from csrl.infrastructure.audio import write_waveform
from csrl.infrastructure.logging import get_logger

_logger = get_logger(__name__)
_HASH_BYTES = 8


class TtsPort(Protocol):
    @property
    def data_source(self) -> DataSource: ...

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        """Mono float32 waveform in [-1, 1] and its sample rate. Raises TtsError."""
        ...


@dataclass(frozen=True)
class SynthesisSettings:
    voices_per_sentence: int
    max_failure_fraction: float
    seed: int


@dataclass(frozen=True)
class Assignment:
    sentence: SentenceRecord
    voice: VoiceSpec

    @property
    def utterance_id(self) -> str:
        separator = constants.SYNTHESIS_ID_SEPARATOR
        voice = self.voice.speaker_id.replace(
            constants.SPEAKER_ID_SEPARATOR, constants.FILENAME_SAFE_SEPARATOR
        )
        return f"{self.sentence.sentence_id}{separator}{voice}"


def sentence_id_of(utterance: Utterance) -> str:
    """Grouping key for synthetic splits: all voices of one sentence stay together."""
    return utterance.utterance_id.split(constants.SYNTHESIS_ID_SEPARATOR)[0]


def plan_assignments(
    sentences: Sequence[SentenceRecord], voices: Sequence[VoiceSpec], settings: SynthesisSettings
) -> list[Assignment]:
    """Deterministic voice choice per sentence, balanced across the pool."""
    if settings.voices_per_sentence > len(voices):
        raise ConfigError(
            f"voices_per_sentence {settings.voices_per_sentence} exceeds pool of {len(voices)}"
        )
    speaker_ids = [v.speaker_id for v in voices]
    if len(set(speaker_ids)) != len(speaker_ids):
        raise ConfigError("voice pool contains duplicate speaker ids")
    plan = []
    for sentence in sentences:
        digest = hashlib.sha256(f"{settings.seed}{sentence.sentence_id}".encode()).digest()
        offset = int.from_bytes(digest[:_HASH_BYTES], "little") % len(voices)
        for k in range(settings.voices_per_sentence):
            plan.append(Assignment(sentence, voices[(offset + k) % len(voices)]))
    return plan


def _completed_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return {u.utterance_id for u in load_manifest(manifest_path)}


def run_synthesis(
    tts: TtsPort,
    plan: Sequence[Assignment],
    output_directory: Path,
    audio_base_directory: Path,
    manifest_path: Path,
    settings: SynthesisSettings,
) -> int:
    """Synthesize every planned item not already in the manifest. Returns items written.

    Manifest audio paths are written relative to audio_base_directory, which must contain
    output_directory. Training configs use the same directory as manifest_base_directory.
    """
    try:
        relative_directory = output_directory.resolve().relative_to(audio_base_directory.resolve())
    except ValueError as error:
        raise ConfigError(
            f"{output_directory} is not inside audio base {audio_base_directory}"
        ) from error
    done = _completed_ids(manifest_path)
    failures_path = output_directory / constants.FAILURES_FILE_NAME
    written = failed = 0
    pending = [a for a in plan if a.utterance_id not in done]
    _logger.info("%d planned, %d already done, %d pending", len(plan), len(done), len(pending))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    for assignment in pending:
        try:
            waveform, sample_rate = tts.synthesize(assignment.sentence.text, assignment.voice)
        except TtsError as error:
            failed += 1
            rejection = Rejection(
                assignment.utterance_id, RejectReason.TTS_FAILURE.value, str(error)
            )
            with failures_path.open("a", encoding=constants.TEXT_ENCODING) as handle:
                handle.write(json.dumps(rejection.__dict__, ensure_ascii=False) + "\n")
            _logger.warning("TTS failure %s: %s", assignment.utterance_id, error)
            if failed / len(pending) > settings.max_failure_fraction:
                raise SynthesisBudgetError(
                    f"{failed} failures out of {len(pending)} pending exceeds "
                    f"{settings.max_failure_fraction}"
                ) from error
            continue
        audio = resample(waveform, sample_rate, constants.SAMPLE_RATE_HZ)
        relative = Path(f"{assignment.utterance_id}{constants.WAV_SUFFIX}")
        write_waveform(output_directory / relative, audio, constants.SAMPLE_RATE_HZ)
        utterance = Utterance(
            utterance_id=assignment.utterance_id,
            audio_path=str(relative_directory / relative),
            duration_seconds=len(audio) / constants.SAMPLE_RATE_HZ,
            speaker_id=assignment.voice.speaker_id,
            text=assignment.sentence.text,
            source=tts.data_source,
        )
        with manifest_path.open("a", encoding=constants.TEXT_ENCODING) as handle:
            handle.write(serialize(utterance) + "\n")
        written += 1
    _logger.info("synthesis finished: %d written, %d failed", written, failed)
    return written
