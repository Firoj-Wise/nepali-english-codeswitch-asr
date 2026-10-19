# 0001 Metric definitions

Status: accepted. Date: 2026-10-19.

## Context

Code-switched Nepali-English output mixes Devanagari and Latin tokens. Plain WER penalizes Nepali agglutination heavily, and plain CER hides English word errors.

## Decision

Use MER (Devanagari as characters, Latin as words) as the primary accuracy metric, plus SCR and SPER as the primary code-switching metrics, as defined in `docs/metrics_spec.md` version 1.

## Consequences

Results are not directly comparable to papers reporting WER only. WER and CER are reported alongside for that reason.
