"""Script confusion rate and switch-point error rate."""

from csrl.domain import constants
from csrl.domain.entities import ConfusionCounts, ErrorCounts
from csrl.domain.enums import EditOperation, Script
from csrl.domain.errors import EmptyReferenceError
from csrl.metrics.alignment import align
from csrl.metrics.error_rates import words
from csrl.metrics.script import token_script


def confusion_counts(reference: str, hypothesis: str) -> ConfusionCounts:
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    reference_scripts = [token_script(w) for w in reference_words]
    confusions = {Script.DEVANAGARI: 0, Script.LATIN: 0}
    for pair in align(reference_words, hypothesis_words):
        if pair.operation is not EditOperation.SUBSTITUTION:
            continue
        if pair.reference_index is None or pair.hypothesis_index is None:
            raise AssertionError("substitution without both indices")
        reference_script = reference_scripts[pair.reference_index]
        hypothesis_script = token_script(hypothesis_words[pair.hypothesis_index])
        if (
            reference_script in constants.CONFUSABLE_SCRIPTS
            and hypothesis_script in constants.CONFUSABLE_SCRIPTS
            and reference_script is not hypothesis_script
        ):
            confusions[reference_script] += 1
    return ConfusionCounts(
        confusions_ne=confusions[Script.DEVANAGARI],
        confusions_en=confusions[Script.LATIN],
        eligible_ne=reference_scripts.count(Script.DEVANAGARI),
        eligible_en=reference_scripts.count(Script.LATIN),
    )


def scr(counts: ConfusionCounts) -> float:
    eligible = counts.eligible_ne + counts.eligible_en
    if eligible == 0:
        raise EmptyReferenceError("no Devanagari or Latin reference words for SCR")
    return (counts.confusions_ne + counts.confusions_en) / eligible


def switch_window_indices(reference_words: list[str]) -> set[int]:
    scripts = [token_script(w) for w in reference_words]
    window: set[int] = set()
    for index in range(1, len(scripts)):
        previous, current = scripts[index - 1], scripts[index]
        if (
            previous in constants.CONFUSABLE_SCRIPTS
            and current in constants.CONFUSABLE_SCRIPTS
            and previous is not current
        ):
            window.update({index - 1, index})
    return window


def switch_point_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Errors on reference words inside switch windows (docs/metrics_spec.md section 6)."""
    reference_words = words(reference)
    window = switch_window_indices(reference_words)
    substitutions = deletions = insertions = 0
    last_reference_index = -1
    for pair in align(reference_words, words(hypothesis)):
        if pair.reference_index is not None:
            last_reference_index = pair.reference_index
        in_window_reference = pair.reference_index in window
        if pair.operation is EditOperation.SUBSTITUTION and in_window_reference:
            substitutions += 1
        elif pair.operation is EditOperation.DELETION and in_window_reference:
            deletions += 1
        elif pair.operation is EditOperation.INSERTION and last_reference_index in window:
            insertions += 1
    return ErrorCounts(substitutions, deletions, insertions, len(window))
