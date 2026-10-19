# RL Objectives

All variants are implemented in `src/csrl/application/grpo.py` and selected in `configs/grpo.yaml` under `objective`. Changing the objective is an experiment, not a fix: register a new run id for each variant.

## Variants and their settings

| Variant | advantage_normalization | loss_aggregation | importance_ratio | clip_low | clip_high | kl_coefficient | dynamic_sampling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | group_std | sequence_mean | token | 0.2 | 0.2 | positive, for example 0.04 | false |
| DAPO (default) | group_std | token_mean | token | 0.2 | 0.28 | 0.0 | true |
| Dr. GRPO | group_mean | constant_normalizer | token | 0.2 | 0.2 | 0.0 | false |
| GSPO | group_std | sequence_mean | sequence | small, see note | small, see note | 0.0 | false |

Notes:

1. GSPO computes one length-normalized importance ratio per sequence, exp of the mean token log ratio, and clips it at the sequence level. Its clip ranges are far smaller than token-level ranges. Take the values from the GSPO paper and record them in the run config. The config loader rejects GSPO with any aggregation other than `sequence_mean`.
2. Clipping has no effect with `update_epochs: 1`, because the importance ratio is exactly one on the first pass. Use `update_epochs` of 2 or more when comparing clip settings.
3. `dynamic_sampling: true` drops groups whose rewards are all equal, because their advantages are zero. DAPO refills the batch by resampling. This repository drops without refilling, so the effective batch size varies. The count is logged as `zero_variance_groups`.
4. With `kl_coefficient: 0.0` no reference model is built, which saves memory. KL is then logged as 0.
5. Dropout is disabled during sampling and updates, so that the sampled and the updated distributions match.
6. Sampling and log probabilities use the same temperature and the same optional gate bias.

## What this repository does not implement

The following parts of DAPO are deliberately absent:

- Overlong reward shaping. Transcripts are short and bounded by `max_new_tokens`.
- Resample-to-fill dynamic sampling.

Any addition needs a decision record.

## Recommended ablation order

1. DAPO default (S3 and S4 in `docs/research_plan.md`).
2. GRPO with a KL term, as the classic baseline.
3. Dr. GRPO, to test whether std normalization biases toward easy utterances.
4. GSPO, if time remains.

Compare variants only at equal sample budgets (batch_size times group_size times steps).

## Sources

Each link below was checked to exist on 2026-10-19. The descriptions of the methods come from the papers as known before that date. An agent must reread the paper before citing a specific number from it.

- DAPO, arXiv 2503.14476: https://arxiv.org/abs/2503.14476
- Understanding R1-Zero-Like Training (Dr. GRPO), arXiv 2503.20783: https://arxiv.org/abs/2503.20783
- Group Sequence Policy Optimization (GSPO), arXiv 2507.18071: https://arxiv.org/abs/2507.18071
