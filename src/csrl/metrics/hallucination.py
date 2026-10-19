"""Hallucination indicators."""

from csrl.domain import constants
from csrl.domain.errors import EmptyReferenceError
from csrl.metrics.error_rates import mixed_units, words


def length_ratio(reference: str, hypothesis: str) -> float:
    reference_length = len(mixed_units(reference))
    if reference_length == 0:
        raise EmptyReferenceError(f"reference is empty: {reference!r}")
    return len(mixed_units(hypothesis)) / reference_length


def has_consecutive_repeat(hypothesis: str) -> bool:
    tokens = words(hypothesis)
    size = constants.REPEAT_NGRAM_SIZE
    for start in range(len(tokens)):
        ngram = tokens[start : start + size]
        if len(ngram) < size:
            break
        repeats = 1
        cursor = start + size
        while tokens[cursor : cursor + size] == ngram:
            repeats += 1
            cursor += size
        if repeats >= constants.REPEAT_MIN_COUNT:
            return True
    return False


def is_hallucination(reference: str, hypothesis: str) -> bool:
    return (
        has_consecutive_repeat(hypothesis)
        or length_ratio(reference, hypothesis) > constants.HALLUCINATION_LENGTH_RATIO
    )
