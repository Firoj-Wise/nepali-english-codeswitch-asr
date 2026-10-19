"""Round-trip quality filter for synthetic audio using an independent ASR.

The filter ASR must not be the model under study. Filtering with indic-transcribe-flex
would keep only audio that model already handles and bias every result.

A Nepali-only ASR writes English words in Devanagari, which would make every
code-switched utterance fail a full-text comparison. With scope devanagari_only, CER is
computed on the Devanagari words of the reference against the Devanagari words of the
hypothesis, counting only substitutions and deletions of reference characters. Insertions
are not counted, because the ASR may render the English words in Devanagari. A speaking
rate check (characters per second) catches truncated, padded, or babbled audio, including
the English part. With scope full, the ordinary CER is used.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from csrl.data.normalize import normalize_text
from csrl.domain import constants
from csrl.domain.entities import Rejection, Utterance
from csrl.domain.enums import RejectReason, RoundTripScope, Script
from csrl.domain.interfaces import Transcriber
from csrl.infrastructure.logging import get_logger
from csrl.metrics.error_rates import cer_counts, rate, words
from csrl.metrics.script import token_script

_logger = get_logger(__name__)


@dataclass(frozen=True)
class FilterSettings:
    max_round_trip_cer: float
    min_duration_seconds: float
    max_duration_seconds: float
    scope: RoundTripScope
    min_characters_per_second: float
    max_characters_per_second: float


def scoped_text(text: str, scope: RoundTripScope) -> str:
    normalized = normalize_text(text)
    if scope is RoundTripScope.FULL:
        return normalized
    return constants.TOKEN_SEPARATOR.join(
        w for w in words(normalized) if token_script(w) is Script.DEVANAGARI
    )


@dataclass(frozen=True)
class FilterResult:
    kept: list[Utterance]
    rejected: list[Rejection]


def filter_synthetic(
    utterances: Sequence[Utterance], transcriber: Transcriber, settings: FilterSettings
) -> FilterResult:
    kept: list[Utterance] = []
    rejected: list[Rejection] = []
    in_range: list[Utterance] = []
    for utterance in utterances:
        if utterance.duration_seconds < settings.min_duration_seconds:
            rejected.append(Rejection(utterance.utterance_id, RejectReason.DURATION_SHORT.value,
                                      f"{utterance.duration_seconds:.2f}"))
        elif utterance.duration_seconds > settings.max_duration_seconds:
            rejected.append(Rejection(utterance.utterance_id, RejectReason.DURATION_LONG.value,
                                      f"{utterance.duration_seconds:.2f}"))
        else:
            characters = len(normalize_text(utterance.text).replace(constants.TOKEN_SEPARATOR, ""))
            speaking_rate = characters / utterance.duration_seconds
            if not (settings.min_characters_per_second <= speaking_rate
                    <= settings.max_characters_per_second):
                rejected.append(Rejection(utterance.utterance_id,
                                          RejectReason.SPEAKING_RATE.value,
                                          f"{speaking_rate:.1f} chars/s"))
            else:
                in_range.append(utterance)
    hypotheses = transcriber.transcribe(in_range)
    for utterance, hypothesis in zip(in_range, hypotheses, strict=True):
        reference = scoped_text(utterance.text, settings.scope)
        if not reference:
            rejected.append(Rejection(utterance.utterance_id, RejectReason.EMPTY.value,
                                      f"no text in scope {settings.scope.value}"))
            continue
        counts = cer_counts(reference, scoped_text(hypothesis, settings.scope))
        if settings.scope is RoundTripScope.DEVANAGARI_ONLY:
            cer = (counts.substitutions + counts.deletions) / counts.reference_length
        else:
            cer = rate(counts)
        if cer > settings.max_round_trip_cer:
            rejected.append(Rejection(utterance.utterance_id, RejectReason.ROUND_TRIP_CER.value,
                                      f"cer={cer:.3f} hyp={hypothesis}"))
        else:
            kept.append(utterance)
    _logger.info("filter kept %d of %d", len(kept), len(utterances))
    return FilterResult(kept, rejected)
