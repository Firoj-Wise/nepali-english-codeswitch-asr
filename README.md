# csrl: Script-Aware Reinforcement Learning for Nepali-English Code-Switched ASR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Firoj112/nepali-english-codeswitch-podcasts)

Code, experiment registry, and evaluation harness for reducing script confusion and switch-point errors in Nepali-English code-switched automatic speech recognition (ASR).

Training uses 100 percent synthetic code-switched speech synthesized via a dual-engine architecture (OmniVoice and Gemini TTS). All claims are evaluated on spontaneous human conversational speech from public Nepali podcasts and verified with paired bootstrap confidence intervals.

---

## Overview

Nepali speakers frequently interleave English lexical items into Nepali conversational speech. Standard multilingual ASR systems often confuse scripts (writing English loanwords in Devanagari script or Nepali words in Latin script) and exhibit high error rates at linguistic switch points.

This repository provides:
1. **Benchmark dataset**: 12.80 hours of verified spontaneous Nepali-English code-switched podcast speech (10,214 test utterances, 2,830 dev utterances) released on Hugging Face at [`Firoj112/nepali-english-codeswitch-podcasts`](https://huggingface.co/datasets/Firoj112/nepali-english-codeswitch-podcasts).
2. **Two-stage adaptation methodology**: Supervised Fine-Tuning (SFT) on synthetic speech, followed by sequence-level Group Relative Policy Optimization (GRPO) with a joint Mixed Error Rate (MER) and Script Confusion Rate (SCR) reward.
3. **Reproducibility**: Every number reported in the research paper is directly traceable to run artifacts in `experiments/` and verified in `experiments/registry.md`.

---

## Primary Results

Evaluations conducted on the frozen real test split (`data/splits/real_test.jsonl`, 10,214 utterances, 12.80 hours, SHA256 verified) using metric version 1:

| System | Run ID | Split | MER (%) | WER (%) | CER (%) | SPER (%) | SCR English (%) | Overall SCR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0 Base** (Zero-Shot) | `S0-20261020-real-test-run01` | test | 26.96 | 44.04 | 23.80 | 41.63 | 8.02 | 3.94 |
| **S1 SFT** (Synthetic) | `S1-20261020-real-test-run01` | test | 24.60 | 40.86 | 21.88 | 36.24 | 7.54 | 3.62 |
| **S4 GRPO** (Script-Aware RL) | `S4-20261020-real-test-run01` | test | **24.57** | **40.81** | **21.84** | **36.17** | **7.44** | **3.60** |

*Definitions*: MER (Mixed Error Rate: Word Error Rate on Latin script, Character Error Rate on Devanagari script); SPER (Switch-Point Error Rate); SCR (Script Confusion Rate).

### Paired Bootstrap Significance (10,000 Resamples, Seed 1234)

| Comparison | Metric | Point Estimate | 95% Confidence Interval | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: |
| **S1 vs S0** | MER | -2.36% | [-3.08%, -1.70%] | Yes (p < 0.05) |
| **S1 vs S0** | SPER | -5.38% | [-6.39%, -4.68%] | Yes (p < 0.05) |
| **S4 vs S0** | MER | -2.40% | [-3.11%, -1.74%] | Yes (p < 0.05) |
| **S4 vs S0** | SPER | -5.46% | [-6.46%, -4.74%] | Yes (p < 0.05) |
| **S4 vs S0** | SCR English | -0.58% | [-0.86%, -0.30%] | Yes (p < 0.05) |
| **S4 vs S1** | SPER | -0.07% | [-0.12%, -0.03%] | Yes (p < 0.05) |
| **S4 vs S1** | SCR English | -0.10% | [-0.13%, -0.06%] | Yes (p < 0.05) |

---

## Monolingual Retention (Zero Catastrophic Forgetting)

To verify that synthetic code-switched adaptation preserves monolingual capabilities, models were evaluated on standard monolingual benchmarks:

| Benchmark | Language | Split | S0 Base | S4 GRPO (Ours) | Delta | 95% Confidence Interval | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BHASAANUVAD** | Nepali | 1,000 utts (1.95h) | 20.80% MER | **18.42% MER** | -2.38% | [-3.99%, -0.74%] | Statistically significant gain (p < 0.05) |
| **LibriSpeech** | English | 1,000 utts (1.50h) | 3.33% WER | **3.38% WER** | +0.05% | [-0.17%, +0.30%] | Zero degradation (p > 0.05) |

---

## Ablation Study (Development Set, 2,830 Utterances)

Evaluating the contribution of supervised fine-tuning, auxiliary token-level script gating, and sequence-level policy optimization:

| Objective | System | Run ID | MER (%) | SPER (%) | SCR English (%) | Observations |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Zero-Shot Base** | S0 | `S0-20261020-real-dev-run01` | 21.36 | 38.09 | 6.95 | Baseline unadapted model |
| **Synthetic SFT** | S1 | `S1-20261020-real-dev-run01` | 19.82 | 32.28 | 6.46 | Cross-entropy adaptation on synthetic data |
| **SFT + Script Gate** | S2 | `S2-20261020-real-dev-run01` | 20.04 | 32.38 | 7.41 | Linear classifier logit biasing introduces slight boundary rigidity |
| **Script-Aware GRPO** | S4 | `S4-20261020-real-dev-run01` | **19.80** | **32.18** | **6.36** | Joint reward optimizes sequence-level script fidelity |
| **GRPO + Script Gate** | S5 | `S5-20261020-real-dev-run01` | 20.02 | 32.35 | 7.39 | Logit gate restricts RL exploration at boundaries |

---

## Base Model & Architecture

- **Backbone**: `bodhan-ai/indic-transcribe-flex` (FastConformer encoder, Transformer decoder, 1.6B parameters, 7,152-token SentencePiece vocabulary).
- **Adaptation**: Frozen encoder, LoRA rank 16 on decoder query and value projections (`linear_q`, `linear_v`).
- **RL Algorithm**: Group Relative Policy Optimization (GRPO) with group size K=8, temperature 0.8, and length-normalized mixed reward:
  $$R_{\text{joint}} = R_{\text{mer}} + \alpha R_{\text{scr}}$$
  where $R_{\text{mer}} = 1 - \text{MER}$ and $R_{\text{scr}} = 1 - \text{SCR}$.

---

## Repository Structure

```
├── configs/                     # Experiment hyperparameters and model configurations
│   ├── model.yaml               # IndicTranscribeFlex backbone and LoRA settings
│   ├── sft.yaml                 # Supervised fine-tuning configuration
│   ├── grpo.yaml                # Script-aware GRPO policy optimization settings
│   └── data.yaml                # Split manifests and dataset parameters
├── src/csrl/                    # Core library implementation
│   ├── domain/                  # Constants, entities, and typed exceptions
│   ├── metrics/                 # Pure metric functions (MER, SPER, SCR, bootstrap)
│   ├── data/                    # Audio loading, text normalization, manifest parsing
│   ├── infrastructure/          # Logging and configuration loaders
│   ├── application/             # Training loops, GRPO policy, and evaluation runners
│   └── adapters/                # Model bindings for IndicTranscribeFlex
├── scripts/                     # Command line tools and verification utilities
│   ├── verify_numbers.py        # Validates registry numbers against test splits
│   ├── run_eval.py              # Evaluates checkpoints across manifests
│   ├── train_sft.py             # SFT training pipeline
│   ├── train_grpo.py            # GRPO policy optimization pipeline
│   └── prepare_splits.py        # Split creation with speaker disjointness
├── experiments/
│   └── registry.md              # Append-only registry of all benchmark runs
├── reports/
│   └── 20261020_synthetic_rl_codeswitch_nepali.md  # Formal research report
└── pyproject.toml               # Package dependencies and build configuration
```

---

## Quickstart

### 1. Installation

Ensure Python 3.10 or higher is installed:

```bash
git clone https://github.com/Firojpaudel/nepali-english-codeswitch-asr.git
cd nepali-english-codeswitch-asr

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,train,data]"

# Or using standard pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train,data]"
```

### 2. Verify Benchmark Numbers

To verify that the reported numbers match the evaluation outputs:

```bash
python scripts/verify_numbers.py
```

### 3. Evaluate a Checkpoint

To evaluate a model checkpoint on the real test split:

```bash
python scripts/run_eval.py \
    --run-id eval-s4-test \
    --system S4 \
    --model-config configs/model.yaml \
    --split test \
    --manifest data/splits/real_test.jsonl \
    --audio-base data \
    --decoder official
```

### 4. Run SFT and GRPO Training

```bash
# Stage 1: Supervised Fine-Tuning
python scripts/train_sft.py \
    --run-id S1-sft-run \
    --system S1 \
    --config configs/sft.yaml \
    --model-config configs/model.yaml

# Stage 2: Script-Aware GRPO Policy Optimization
python scripts/train_grpo.py \
    --run-id S4-grpo-run \
    --system S4 \
    --config configs/grpo.yaml \
    --model-config configs/model.yaml
```

---

## Dataset Release

The evaluation benchmark audio and transcriptions are available on the Hugging Face Hub:

- **Repository**: [`Firoj112/nepali-english-codeswitch-podcasts`](https://huggingface.co/datasets/Firoj112/nepali-english-codeswitch-podcasts)
- **Splits**: `dev` (2,830 rows, 3.55 hours) and `test` (10,214 rows, 12.80 hours).
- **Audio format**: 16 kHz mono WAV / FLAC, normalized duration.
- **Audit documentation**: `data/audit/convention_violations.jsonl` catalogs flagged acoustic repetition segments.

---

## Citation

If you use this benchmark or codebase in your research, please cite:

```bibtex
@misc{paudel2026csrl,
  author = {Paudel, Firoj},
  title = {Reducing Script Confusion and Switch-Point Errors in Nepali-English Code-Switched ASR via Synthetic SFT and Reinforcement Learning},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Firojpaudel/nepali-english-codeswitch-asr}}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
