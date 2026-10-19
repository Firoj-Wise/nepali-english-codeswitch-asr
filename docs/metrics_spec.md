# Metrics Specification

Metric version: 1. Any change to this file requires a decision record and a bump of `METRIC_VERSION`.

All metrics are computed by `src/csrl/metrics/`. No other code computes reported numbers.

## 1. Normalization (applied identically to reference and hypothesis)

Applied in this order by `csrl.data.normalize.normalize_text`:

1. Unicode NFC normalization.
2. Lowercase Latin characters. Devanagari is unchanged.
3. Remove punctuation listed in `constants.PUNCTUATION_TO_STRIP`, including the Devanagari danda and double danda.
4. Map Devanagari digits to ASCII digits.
5. Insert a space at every boundary between a Latin letter or digit and a Devanagari character, in both directions. Example: "phoneमा" becomes "phone मा". Writers attach Nepali postpositions to English words inconsistently, and this step makes all such variants score the same. See `docs/decisions/0002-script-boundary-split.md`.
6. Collapse runs of whitespace to one space and strip ends.

Numbers are not verbalized. The base model mixed mode applies inverse text normalization, so a mismatch such as "5" versus "पाँच" is counted as an error and reported separately as `itn_mismatch_count` (see section 6) so it can be excluded from interpretation.

## 2. Token script labels

Each whitespace token receives one label from `enums.Script`:

- `DEVANAGARI`: every letter character is in the Devanagari block (U+0900 to U+097F).
- `LATIN`: every letter character is Latin.
- `NUMERIC`: no letters, at least one digit.
- `MIXED`: letters from more than one script. Counted, never silently merged.
- `OTHER`: none of the above.

## 3. Mixed Error Rate (MER)

MER is the error rate over a token sequence in which Devanagari tokens are split into characters and Latin and numeric tokens remain words. This follows common code-switched ASR practice of character-level scoring for the Indic side and word-level scoring for English.

MER = (S + D + I) / N, where S, D, I are substitutions, deletions, and insertions from a Levenshtein alignment of the mixed unit sequences, and N is the number of reference units. Corpus MER sums S, D, I, N over all utterances before dividing. It is never an average of per-utterance rates.

If N is zero for an utterance, that utterance is rejected with `EmptyReferenceError`. It is never scored as zero.

## 4. WER and CER

Reported alongside MER for comparability. WER uses whitespace tokens. CER uses characters with spaces removed. Corpus-level aggregation as in section 3.

## 5. Script Confusion Rate (SCR)

Align reference and hypothesis at the word level. For each aligned substitution pair (r, h), a script confusion occurs when `script(r)` and `script(h)` are both in {DEVANAGARI, LATIN} and differ. Example: reference `phone`, hypothesis `फोन`.

SCR = script confusions / reference words whose script is DEVANAGARI or LATIN.

Report SCR overall and separately by reference script (SCR_ne, SCR_en).

## 6. Switch-point error rate (SPER)

A reference switch point is a word index i where the script of word i differs from the script of word i minus 1 (both in {DEVANAGARI, LATIN}). The switch window is words i minus 1 and i. SPER is the word error rate restricted to reference words inside any switch window, counting substitutions and deletions of those words. Insertions are attributed to the nearest reference word to the left.

## 7. Hallucination indicators

- `length_ratio`: hypothesis units divided by reference units, per utterance. Flagged when above `constants.HALLUCINATION_LENGTH_RATIO`.
- `repeat_flag`: any n-gram of size `constants.REPEAT_NGRAM_SIZE` repeated at least `constants.REPEAT_MIN_COUNT` times consecutively.
- `empty_hypothesis_count`: hypotheses empty after normalization. Always reported. A nonzero count must be explained in the report.

## 8. Statistical testing

Paired bootstrap over utterances with `constants.BOOTSTRAP_SAMPLES` resamples and the seed from config. Report the 95 percent confidence interval of the difference between systems. A difference is claimed only when the interval excludes zero. Corpus-level metrics are recomputed inside each resample, not averaged.

## 9. Reward functions (training only, never reported as results)

Rewards reuse the metric functions above so that training and evaluation agree.

- `R_mer = -MER(utterance)`, clipped below at `-constants.REWARD_MER_CLIP`.
- `R_scr = -lambda_scr * script_confusions(utterance)`.
- `R_hal = -lambda_hal * (repeat_flag or length_ratio above threshold)`.
- Total reward is the sum. Lambda values live in `configs/grpo.yaml`.

The reward function is versioned together with `METRIC_VERSION`.
