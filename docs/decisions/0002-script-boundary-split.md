# 0002 Split tokens at script boundaries

Status: accepted. Date: 2026-10-19. Applies from metric version 1; no results existed before this decision.

## Context

Code-switched Nepali attaches Nepali postpositions and suffixes to English words in several ways: "phoneमा", "phone-मा", "phone मा". References, model output, and generated sentences do not agree on one form. Without a rule, the same spoken words would count as different tokens, and attached forms would be labelled MIXED and fall outside script confusion counting.

## Decision

Normalization inserts a space at every Latin-to-Devanagari and Devanagari-to-Latin boundary, digits included, before tokenization (`docs/metrics_spec.md`, step 5). The text generation prompt asks for the separated form directly.

## Consequences

Word counts for code-switched text are slightly higher than an orthographic count. The rule is applied identically to references, hypotheses, rewards, filters, and leakage checks, because they all call `normalize_text`.
