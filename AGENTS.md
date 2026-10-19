# Agent Operating Rules

These rules apply to every agent and every human working in this repository. They override convenience. When a rule and a request conflict, stop and ask.

## 0. What this repository is for

Goal: measure and reduce script confusion and switch-point errors in Nepali-English code-switched ASR, and produce a short paper whose every number is reproducible from `experiments/registry.md`. The full statement is in `README.md`, and the novelty boundary is in `docs/related_work.md`. Work that does not serve this goal is out of scope. Ask before starting it.

## 1. Read order before any work

1. `README.md`
2. `docs/related_work.md`
3. `docs/research_plan.md`
4. `docs/metrics_spec.md`
5. `docs/data_spec.md`
6. `docs/experiment_protocol.md`
7. `docs/rl_objectives.md`
8. `docs/coding_standards.md`
9. `docs/writing_style.md`
10. `experiments/registry.md` (to know what has already been run)

## 2. Integrity rules (non-negotiable)

1. Never report a number that was not produced by code in this repository and recorded in `experiments/`. Every number in a report must cite its run id.
2. Never invent a citation, URL, dataset, model name, or result. Every external reference must already be in `docs/references.md` or be added there with a link you have opened. If you cannot verify it, write `UNVERIFIED` and stop.
3. Never touch the test split except through `csrl.application.evaluate` with a frozen checkpoint. No tuning, no early stopping, no prompt or threshold selection on test. Use dev.
4. Never change a metric definition silently. Metric changes require an entry in `docs/decisions/` and a version bump of `METRIC_VERSION` in `src/csrl/domain/constants.py`. Results computed under different metric versions are not comparable and must not share a table.
5. Never delete or rewrite rows in `experiments/registry.md`. Append only. A failed or invalid run is marked `INVALID` with a reason.
6. If a result looks surprisingly good, assume a bug first. Check for test leakage, normalization mismatch, and empty hypotheses before reporting.
7. Report negative and null results with the same care as positive ones.
8. Every claimed improvement needs a paired bootstrap confidence interval (see `docs/metrics_spec.md`). No interval, no claim.
9. No training run starts until `scripts/smoke_test_model.py` has passed on the current commit of `src/csrl/adapters/indic_transcribe.py`. Record the smoke test result in the registry notes of the first training run.
10. Do not change `prompt_token_ids` by guessing. Derive it from `experiments/model_inspection.txt` and confirm it with the smoke test.
11. Never claim novelty for anything listed as prior work in `docs/related_work.md`. Before writing any novelty claim, search for newer work and update that file.
12. Never put company code, weights, configs, checkpoint paths with internal hostnames, or style tag lists into this repository. Private models are reached only through a path in config and an external style plugin.
13. API keys come only from environment variables. Never write a key into any file.
14. The round-trip filter ASR must be independent of indic-transcribe-flex.
15. Real recordings are evaluation data only. They never enter training, filtering thresholds, or prompt design.
16. If you are unsure, say so in writing. "I do not know" is an acceptable output. A guessed answer is not.

## 3. Code rules (summary; full text in `docs/coding_standards.md`)

1. No magic values in code. Every literal that carries meaning lives in `src/csrl/domain/constants.py`, an enum in `src/csrl/domain/enums.py`, or a config file in `configs/`.
2. No silent fallbacks. No bare `except`, no `except Exception: pass`, no `return None` to signal failure, no default value substituted for missing data. Raise a typed error from `src/csrl/domain/errors.py`.
3. No empty returns. Functions return a typed value or raise.
4. Logging through `csrl.infrastructure.logging` only. No `print`.
5. Clean architecture: `domain` and `metrics` import nothing from `data`, `infrastructure`, or `application`. Dependencies point inward.
6. Every new function in `metrics/` ships with a unit test containing at least one hand-computed example.
7. Run `make check` before declaring any task done. A task with a failing check is not done.

## 4. Writing rules (summary; full text in `docs/writing_style.md`)

1. No em dashes or en dashes. Use commas, colons, parentheses, or a new sentence.
2. No emoji, no decorative symbols, no exclamation marks.
3. No marketing language. The banned list is `scripts/banned_phrases.txt`. "Robust" is allowed only with a stated definition.
4. State numbers with their run id, split, metric version, and confidence interval.
5. `scripts/check_repo.py` enforces the mechanical parts of these rules.

## 5. What to do when blocked

Write the blocker into the task output with: what was attempted, the exact error, and what decision is needed. Do not work around it with a fallback.

## 6. Definition of done for any task

- `make check` passes.
- New behavior has tests.
- Any run is registered in `experiments/registry.md`.
- Any report follows `reports/TEMPLATE.md`.
