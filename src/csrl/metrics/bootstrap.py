"""Paired bootstrap over utterances for corpus-level error rates."""

import random
from collections.abc import Sequence

from csrl.domain import constants
from csrl.domain.entities import ConfidenceInterval, ErrorCounts
from csrl.domain.errors import LengthMismatchError
from csrl.metrics.error_rates import rate


def _corpus_rate(counts: Sequence[ErrorCounts], indices: Sequence[int]) -> float:
    total = ErrorCounts(0, 0, 0, 0)
    for index in indices:
        total = total + counts[index]
    return rate(total)


def paired_bootstrap_difference(
    system_a: Sequence[ErrorCounts],
    system_b: Sequence[ErrorCounts],
    seed: int,
    samples: int = constants.BOOTSTRAP_SAMPLES,
    confidence: float = constants.CONFIDENCE_LEVEL,
) -> ConfidenceInterval:
    """Interval for rate(A) minus rate(B). Negative means A has fewer errors."""
    if len(system_a) != len(system_b):
        raise LengthMismatchError(f"{len(system_a)} versus {len(system_b)} utterances")
    size = len(system_a)
    all_indices = list(range(size))
    point = _corpus_rate(system_a, all_indices) - _corpus_rate(system_b, all_indices)
    generator = random.Random(seed)
    differences = []
    for _ in range(samples):
        indices = [generator.randrange(size) for _ in range(size)]
        differences.append(_corpus_rate(system_a, indices) - _corpus_rate(system_b, indices))
    differences.sort()
    tail = (1 - confidence) / 2
    lower_index = int(tail * samples)
    upper_index = min(samples - 1, int((1 - tail) * samples))
    return ConfidenceInterval(point, differences[lower_index], differences[upper_index])
