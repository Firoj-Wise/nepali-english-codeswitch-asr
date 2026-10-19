"""Evaluation use case. The only path that produces reported numbers."""

from collections.abc import Sequence
from dataclasses import dataclass

from csrl.data.normalize import normalize_text
from csrl.domain import constants
from csrl.domain.entities import ConfusionCounts, ErrorCounts, Utterance
from csrl.domain.enums import MetricName
from csrl.domain.errors import LengthMismatchError
from csrl.domain.interfaces import Transcriber
from csrl.infrastructure.logging import get_logger
from csrl.metrics.code_switch import confusion_counts, scr, switch_point_counts
from csrl.metrics.error_rates import cer_counts, mer_counts, rate, wer_counts
from csrl.metrics.hallucination import is_hallucination

_logger = get_logger(__name__)


@dataclass(frozen=True)
class UtteranceScores:
    utterance_id: str
    reference: str
    hypothesis: str
    mer: ErrorCounts
    wer: ErrorCounts
    cer: ErrorCounts
    sper: ErrorCounts
    confusion: ConfusionCounts
    hallucination: bool


def score_pairs(
    utterances: Sequence[Utterance], hypotheses: Sequence[str]
) -> list[UtteranceScores]:
    if len(utterances) != len(hypotheses):
        raise LengthMismatchError(f"{len(utterances)} utterances, {len(hypotheses)} hypotheses")
    scores = []
    for utterance, raw_hypothesis in zip(utterances, hypotheses):
        reference = normalize_text(utterance.text)
        hypothesis = normalize_text(raw_hypothesis)
        scores.append(
            UtteranceScores(
                utterance_id=utterance.utterance_id,
                reference=reference,
                hypothesis=hypothesis,
                mer=mer_counts(reference, hypothesis),
                wer=wer_counts(reference, hypothesis),
                cer=cer_counts(reference, hypothesis),
                sper=switch_point_counts(reference, hypothesis),
                confusion=confusion_counts(reference, hypothesis),
                hallucination=is_hallucination(reference, hypothesis),
            )
        )
    return scores


def _sum_errors(items: Sequence[ErrorCounts]) -> ErrorCounts:
    total = ErrorCounts(0, 0, 0, 0)
    for item in items:
        total = total + item
    return total


def summarize(scores: Sequence[UtteranceScores]) -> dict[str, float | int]:
    confusion = ConfusionCounts(0, 0, 0, 0)
    for score in scores:
        confusion = confusion + score.confusion
    sper_total = _sum_errors([s.sper for s in scores])
    summary: dict[str, float | int] = {
        MetricName.METRIC_VERSION.value: constants.METRIC_VERSION,
        MetricName.UTTERANCE_COUNT.value: len(scores),
        MetricName.MER.value: rate(_sum_errors([s.mer for s in scores])),
        MetricName.WER.value: rate(_sum_errors([s.wer for s in scores])),
        MetricName.CER.value: rate(_sum_errors([s.cer for s in scores])),
        MetricName.SCR.value: scr(confusion),
        MetricName.EMPTY_HYPOTHESIS_COUNT.value: sum(not s.hypothesis for s in scores),
        MetricName.HALLUCINATION_FLAG_COUNT.value: sum(s.hallucination for s in scores),
    }
    if sper_total.reference_length > 0:
        summary[MetricName.SPER.value] = rate(sper_total)
    else:
        _logger.warning("no switch points in reference set; SPER not reported")
    if confusion.eligible_ne > 0:
        summary[MetricName.SCR_NE.value] = confusion.confusions_ne / confusion.eligible_ne
    if confusion.eligible_en > 0:
        summary[MetricName.SCR_EN.value] = confusion.confusions_en / confusion.eligible_en
    _logger.info("evaluated %d utterances: %s", len(scores), summary)
    return summary


def evaluate(transcriber: Transcriber, utterances: Sequence[Utterance]) -> dict[str, float | int]:
    hypotheses = transcriber.transcribe(utterances)
    return summarize(score_pairs(utterances, hypotheses))
