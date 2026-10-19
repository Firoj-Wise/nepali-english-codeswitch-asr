"""WER, CER, and MER counts on normalized text. Corpus rates sum counts before dividing."""

from csrl.domain import constants
from csrl.domain.entities import ErrorCounts
from csrl.domain.enums import Script
from csrl.domain.errors import EmptyReferenceError
from csrl.metrics.alignment import count_errors
from csrl.metrics.script import token_script


def words(text: str) -> list[str]:
    return [t for t in text.split(constants.TOKEN_SEPARATOR) if t]


def mixed_units(text: str) -> list[str]:
    """Devanagari tokens become characters; all other tokens stay words."""
    units: list[str] = []
    for token in words(text):
        if token_script(token) is Script.DEVANAGARI:
            units.extend(token)
        else:
            units.append(token)
    return units


def _require_nonempty(units: list[str], reference: str) -> None:
    if not units:
        raise EmptyReferenceError(f"reference is empty after normalization: {reference!r}")


def wer_counts(reference: str, hypothesis: str) -> ErrorCounts:
    reference_units = words(reference)
    _require_nonempty(reference_units, reference)
    return count_errors(reference_units, words(hypothesis))


def cer_counts(reference: str, hypothesis: str) -> ErrorCounts:
    reference_units = list(reference.replace(constants.TOKEN_SEPARATOR, ""))
    _require_nonempty(reference_units, reference)
    return count_errors(reference_units, list(hypothesis.replace(constants.TOKEN_SEPARATOR, "")))


def mer_counts(reference: str, hypothesis: str) -> ErrorCounts:
    reference_units = mixed_units(reference)
    _require_nonempty(reference_units, reference)
    return count_errors(reference_units, mixed_units(hypothesis))


def rate(counts: ErrorCounts) -> float:
    if counts.reference_length == 0:
        raise EmptyReferenceError("cannot compute a rate over zero reference units")
    return counts.errors / counts.reference_length
