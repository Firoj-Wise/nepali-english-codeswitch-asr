import pytest

from csrl.data.normalize import normalize_text
from csrl.domain.entities import ErrorCounts
from csrl.domain.enums import Script
from csrl.domain.errors import EmptyReferenceError, LengthMismatchError
from csrl.metrics.bootstrap import paired_bootstrap_difference
from csrl.metrics.code_switch import confusion_counts, scr, switch_point_counts
from csrl.metrics.error_rates import mer_counts, mixed_units, rate, wer_counts
from csrl.metrics.hallucination import has_consecutive_repeat
from csrl.metrics.reward import RewardWeights, compute_reward
from csrl.metrics.script import token_script


def test_token_script_labels() -> None:
    assert token_script("फोन") is Script.DEVANAGARI
    assert token_script("phone") is Script.LATIN
    assert token_script("5") is Script.NUMERIC
    assert token_script("phoneको") is Script.MIXED


def test_wer_hand_computed() -> None:
    # ref 3 words, one substitution: 1 / 3
    counts = wer_counts("a b c", "a x c")
    assert counts == ErrorCounts(1, 0, 0, 3)


def test_mixed_units_split_devanagari_only() -> None:
    assert mixed_units("मेरो phone") == ["म", "े", "र", "ो", "phone"]


def test_mer_hand_computed() -> None:
    # ref units: म े र ो phone (5). hyp: म े र ो फोन -> units म े र ो फ ो न (7)
    # alignment: 4 matches, phone->फ substitution, 2 insertions -> 3 errors / 5
    assert rate(mer_counts("मेरो phone", "मेरो फोन")) == pytest.approx(3 / 5)


def test_script_confusion_counted() -> None:
    # ref words: मेरो(ne) phone(en). phone->फोन is a Latin to Devanagari confusion.
    counts = confusion_counts("मेरो phone", "मेरो फोन")
    assert counts.confusions_en == 1 and counts.confusions_ne == 0
    assert scr(counts) == pytest.approx(1 / 2)


def test_switch_point_window() -> None:
    # switch between index 0 (ne) and 1 (en); window {0,1}; one substitution in window
    counts = switch_point_counts("मेरो phone राम्रो", "मेरो फोन राम्रो")
    assert counts.reference_length == 3  # {0,1} and {1,2}
    assert counts.substitutions == 1


def test_empty_reference_raises() -> None:
    with pytest.raises(EmptyReferenceError):
        wer_counts("", "a")


def test_normalize() -> None:
    assert normalize_text("Hello, World।  ५") == "hello world 5"
    # attached Nepali suffixes and digits are split at script boundaries
    assert normalize_text("phoneमा 5जना") == "phone मा 5 जना"


def test_repeat_detection() -> None:
    assert has_consecutive_repeat("a b c a b c a b c")
    assert not has_consecutive_repeat("a b c d e f")


def test_reward_perfect_hypothesis_is_zero() -> None:
    reward = compute_reward("मेरो phone", "मेरो phone", RewardWeights(0.5, 1.0))
    assert reward.total == pytest.approx(0.0)


def test_bootstrap_identical_systems() -> None:
    counts = [ErrorCounts(1, 0, 0, 4), ErrorCounts(0, 1, 0, 3)]
    interval = paired_bootstrap_difference(counts, counts, seed=0, samples=50)
    assert interval.point_estimate == 0 and not interval.excludes_zero


def test_bootstrap_length_mismatch() -> None:
    with pytest.raises(LengthMismatchError):
        paired_bootstrap_difference([ErrorCounts(0, 0, 0, 1)], [], seed=0)
