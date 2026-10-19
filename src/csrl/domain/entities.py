"""Plain data entities."""

from dataclasses import dataclass

from csrl.domain.constants import SPEAKER_ID_SEPARATOR
from csrl.domain.enums import DataSource, EditOperation


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    audio_path: str
    duration_seconds: float
    speaker_id: str
    text: str
    source: DataSource


@dataclass(frozen=True)
class AlignedPair:
    operation: EditOperation
    reference_index: int | None
    hypothesis_index: int | None


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )


@dataclass(frozen=True)
class ConfusionCounts:
    confusions_ne: int
    confusions_en: int
    eligible_ne: int
    eligible_en: int

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            self.confusions_ne + other.confusions_ne,
            self.confusions_en + other.confusions_en,
            self.eligible_ne + other.eligible_ne,
            self.eligible_en + other.eligible_en,
        )


@dataclass(frozen=True)
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    text: str
    domain: str
    switch_type: str


@dataclass(frozen=True)
class VoiceSpec:
    """One synthetic speaker. voice_id is a prebuilt voice name or a reference clip id."""

    system: str
    voice_id: str
    style: str
    reference_audio: str | None
    reference_text: str | None

    @property
    def speaker_id(self) -> str:
        separator = SPEAKER_ID_SEPARATOR
        return f"{self.system}{separator}{self.voice_id}{separator}{self.style}"


@dataclass(frozen=True)
class Rejection:
    item_id: str
    reason: str
    detail: str
