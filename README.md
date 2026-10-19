# csrl: Script-Aware Reinforcement Learning for Nepali-English Code-Switched ASR

## Purpose and end goal

Nepali speakers mix English words into Nepali speech. ASR systems often write those English words in Devanagari, or Nepali words in Latin script, and they fail most often at the points where the language switches. This repository exists to measure and reduce those errors.

Training data is synthetic. Every claim is measured on real human speech.

End goal: a short research paper with five parts.

1. A real Nepali-English code-switched evaluation set, with exact metric definitions.
2. An ablation of synthetic data design: number of voices, number of TTS systems, expressive style, and telephony augmentation, each measured on real speech.
3. A controlled comparison of SFT, a token-level script gate, and RL with script-aware rewards, on a strong encoder-decoder ASR model.
4. A switch-point error analysis.
5. Released code that reproduces every number from the experiment registry.

Prior work on RL for code-switched ASR exists and must be cited. See `docs/related_work.md` for what is already done and what this project claims beyond it.

## Research question

Does policy-gradient fine-tuning with a script-aware, sequence-level reward reduce language and script confusion errors in Nepali-English code-switched speech recognition, compared with supervised fine-tuning on the same data? Does an explicit token-level script gate in the decoder change that result?

## Hypotheses

- H1. RL with a reward of negative mixed error rate (MER) lowers MER relative to the SFT checkpoint on the held-out code-switched test split.
- H2. Adding a script-confusion penalty to the reward lowers the script confusion rate (SCR) beyond the MER-only reward, without raising MER. A related reward was shown to help an audio-language model on synthetic speech (arXiv 2607.02757); here it is tested on an encoder-decoder ASR model with real Nepali-English speech.
- H3. A token-level script gate trained with an auxiliary loss lowers SCR at switch points, and its effect is additive with H2.
- H0 for each: no difference outside the paired bootstrap 95 percent confidence interval. A null result is reported as a null result.

## Base model

`bodhan-ai/indic-transcribe-flex` (FastConformer encoder, Transformer decoder, built on `nvidia/canary-1b-v2`), in mixed-script mode with language code `ne`. The encoder is frozen. The decoder, output head, and optional script gate are trained.

## Pipeline

All commands run from the repository root. Data steps 1 to 4 need no GPU except the in-house TTS and the filter ASR.

```
pip install -e ".[dev,train,data]"
make check

# 1. Text: generate with docs/prompts/text_generation_prompt.md, then validate
python scripts/validate_sentences.py --config configs/synthesis.yaml
python scripts/check_leakage.py --synthesis-sentences data/text/sentences.validated.jsonl \
    --recording-sentences data/text/reading_set.jsonl --ngram-size 5

# 2. Speech: pilot with --limit 50 first and listen, then the full run per system and style
export GEMINI_API_KEY=...
python scripts/synthesize.py --config configs/synthesis.yaml --system gemini --style neutral
python scripts/synthesize.py --config configs/synthesis.yaml --system inhouse_omnivoice --style neutral

# 3. Quality filter (independent ASR) and optional telephony arm
python scripts/filter_synthetic.py --config configs/synthesis.yaml \
    --manifest data/manifests/synth_gemini_neutral.jsonl
python scripts/augment_telephony.py --config configs/synthesis.yaml \
    --manifest data/manifests/synth_gemini_neutral.filtered.jsonl

# 4. Splits: synthetic train and dev, real dev and test, leakage check, test freeze
python scripts/prepare_splits.py --config configs/data.yaml

# 5. Model binding: inspect, fill prompt_token_ids in configs/model.yaml, smoke test
python scripts/inspect_model.py --model-config configs/model.yaml
python scripts/smoke_test_model.py --model-config configs/model.yaml \
    --train-config configs/sft.yaml --manifest data/splits/real_dev.jsonl --audio-base data

# 6. S0 zero-shot baseline on real dev
python scripts/run_eval.py --run-id S0-<date>-zeroshot --system S0 \
    --model-config configs/model.yaml --split dev --manifest data/splits/real_dev.jsonl \
    --audio-base data --decoder official

# 7. SFT, then RL (set init_checkpoint in configs/grpo.yaml to the SFT best.pt)
python scripts/train_sft.py --run-id S1-<date>-sft-D4 --system S1 \
    --config configs/sft.yaml --model-config configs/model.yaml
python scripts/train_grpo.py --run-id S4-<date>-dapo-scr --system S4 \
    --config configs/grpo.yaml --model-config configs/model.yaml
```

Every run writes to `experiments/<run id>/` and appends rows to `experiments/registry.md`.

## Repository layout

```
AGENTS.md                  Rules for any agent working in this repository. Read first.
docs/                      Plan, related work, metric, data, protocol, RL objective, code and style specs.
configs/                   All tunable values. Code reads values from here.
src/csrl/
  domain/                  Enums, constants, entities, errors. Standard library only.
  metrics/                 Pure metric and reward functions.
  data/                    Manifests, normalization, speaker-disjoint splits.
  infrastructure/          Logging, config loading, audio loading.
  application/             Policy, script gate, RL objectives, training loops, checkpoints.
  adapters/                Model bindings and the composition root.
scripts/                   Command line entry points and the repository harness.
tests/                     Unit and end-to-end tests on a toy backbone.
experiments/registry.md    One row per run event. Append only.
reports/TEMPLATE.md        Template for every results report.
```

## Verification status

| Component | Status |
| --- | --- |
| Metrics, rewards, bootstrap | Unit tested with hand-computed values |
| RL objectives (GRPO, DAPO, Dr. GRPO, GSPO) | Unit tested, including clipping and gradients |
| Policy decoding, SFT and RL loops | End-to-end tested on a toy backbone |
| Official transcriber (S0) | Uses the documented model card API |
| NeMo binding for training | UNVERIFIED until `scripts/smoke_test_model.py` passes |
