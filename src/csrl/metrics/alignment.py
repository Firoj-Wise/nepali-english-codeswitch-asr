"""Levenshtein alignment with a deterministic backtrace."""

from collections.abc import Sequence
from typing import TypeVar

from csrl.domain.entities import AlignedPair, ErrorCounts
from csrl.domain.enums import EditOperation

T = TypeVar("T")

_UNIT_COST = 1


def align(reference: Sequence[T], hypothesis: Sequence[T]) -> list[AlignedPair]:
    """Align two sequences. Tie order on backtrace: match or substitution, deletion, insertion."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = cost[i - 1][j - 1] + (
                0 if reference[i - 1] == hypothesis[j - 1] else _UNIT_COST
            )
            cost[i][j] = min(diagonal, cost[i - 1][j] + _UNIT_COST, cost[i][j - 1] + _UNIT_COST)

    pairs: list[AlignedPair] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            is_match = reference[i - 1] == hypothesis[j - 1]
            diagonal = cost[i - 1][j - 1] + (0 if is_match else _UNIT_COST)
            if cost[i][j] == diagonal:
                operation = EditOperation.MATCH if is_match else EditOperation.SUBSTITUTION
                pairs.append(AlignedPair(operation, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + _UNIT_COST:
            pairs.append(AlignedPair(EditOperation.DELETION, i - 1, None))
            i -= 1
            continue
        pairs.append(AlignedPair(EditOperation.INSERTION, None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def count_errors(reference: Sequence[T], hypothesis: Sequence[T]) -> ErrorCounts:
    pairs = align(reference, hypothesis)
    return ErrorCounts(
        substitutions=sum(p.operation is EditOperation.SUBSTITUTION for p in pairs),
        deletions=sum(p.operation is EditOperation.DELETION for p in pairs),
        insertions=sum(p.operation is EditOperation.INSERTION for p in pairs),
        reference_length=len(reference),
    )
