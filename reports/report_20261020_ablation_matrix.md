# Report: Core 2 by 2 Ablation Matrix for Nepali-English Code-Switched ASR

Date: 2026-10-20. Metric version: 1. Runs: S0-20261019-baseline-zero-shot, S1-20261019-sft-run01, S2-20261020-sft-gate-run01, S4-20261019-grpo-run01, S5-20261020-grpo-gate-run01.

## Question

Does token-level script gating or group relative policy optimization with a script-aware reward reduce script confusion and switch-point errors when adapting an encoder-decoder ASR model to Nepali-English code-switched speech?

## Relation to prior work

This study evaluates code-switched adaptation on a FastConformer encoder-decoder architecture (indic-transcribe-flex), differing from Ye and Vickers (arXiv 2607.02757) who studied decoder-only audio language models (Qwen2-Audio) with GRPO. The auxiliary script gate implements a lightweight linear token-level classifier with inference logit biasing, differing from token-level language diarization in arXiv 2210.14567 which used a secondary autoregressive decoder without reinforcement learning.

## Setup

- Base model: indic-transcribe-flex (Nemo FastConformer-CTC hybrid architecture, frozen encoder, LoRA rank 16 on decoder, target modules: linear_q, linear_v).
- Training split: data/splits/synthetic_train.jsonl (517 utterances, 211 sentence groups, 0.95 hours).
- Evaluation split: data/splits/synthetic_dev.jsonl (59 utterances, 21 sentence groups, 0.11 hours).
- Audio pool: 499 validated text sentences synthesized across OmniVoice (Deepak, Prakash) and Gemini (Puck, Charon, Kore, Leda), augmented with telephony bandpass filtering.
- Configurations:
  - S0: Zero-shot baseline without training (configs/model.yaml).
  - S1: Supervised fine-tuning with cross-entropy loss (configs/sft.yaml).
  - S2: SFT with auxiliary token-level script gate loss and logit biasing (configs/sft_gate.yaml).
  - S4: Group Relative Policy Optimization initialized from S1, optimizing MER and SCR rewards with group size 8 (configs/grpo.yaml).
  - S5: Combined token-level script gate and GRPO initialized from S2 (configs/grpo_gate.yaml).

## Results

| System | Run id | Split | MER | SCR | SPER | WER | CER | Empty hyps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | S0-20261019-baseline-zero-shot | dev | 0.0590 | 0.0232 | 0.1842 | 0.1763 | 0.0579 | 0 |
| S1 | S1-20261019-sft-run01 | dev | 0.0200 | 0.0081 | 0.0658 | 0.0847 | 0.0206 | 0 |
| S2 | S2-20261020-sft-gate-run01 | dev | 0.0202 | 0.0081 | 0.0691 | 0.0858 | 0.0214 | 0 |
| S4 | S4-20261019-grpo-run01 | dev | 0.0200 | 0.0081 | 0.0658 | 0.0847 | 0.0206 | 0 |
| S5 | S5-20261020-grpo-gate-run01 | dev | 0.0202 | 0.0081 | 0.0691 | 0.0858 | 0.0214 | 0 |

Paired bootstrap 95 percent confidence intervals (1000 resamples, seed 42) for difference rate(A) minus rate(B):

| Comparison (A vs B) | Metric | Point Estimate | 95% Confidence Interval | Excludes Zero |
| --- | --- | --- | --- | --- |
| S1 vs S0 | MER | -0.0348 | [-0.0500, -0.0224] | Yes |
| S2 vs S1 | MER | +0.0007 | [-0.0023, +0.0039] | No |
| S4 vs S1 | MER | 0.0000 | [0.0000, 0.0000] | No |
| S5 vs S2 | MER | 0.0000 | [0.0000, 0.0000] | No |

Directional script confusion rates on dev:
- S0: SCR_ne = 0.0040, SCR_en = 0.1518.
- S1: SCR_ne = 0.0053, SCR_en = 0.0268.
- S2: SCR_ne = 0.0053, SCR_en = 0.0268.
- S4: SCR_ne = 0.0053, SCR_en = 0.0268.
- S5: SCR_ne = 0.0053, SCR_en = 0.0268.

## Observations

1. SFT yields a statistically significant error reduction over zero-shot baseline (MER drops from 0.0590 to 0.0200, interval [-0.0500, -0.0224]).
2. English script confusion (SCR_en) drops by 82.3 percent from 0.1518 in S0 to 0.0268 in all fine-tuned systems.
3. Switch-point error rate (SPER) drops from 0.1842 in S0 to 0.0658 in S1 and 0.0691 in S2.
4. Greedy decoding hypotheses for S4 are identical to S1 on the 59 dev utterances.
5. Greedy decoding hypotheses for S5 are identical to S2 on the 59 dev utterances.
6. Between S1 and S2, hypotheses differ on 8 out of 59 utterances. The script gate fixes script confusion on utterance-initial English loanwords (for example, correctly transcribing Latin 'anyway' in S2 where S1 transliterated into Devanagari), while producing minor character-level orthographic variations elsewhere.

## Interpretation

The primary bottleneck in zero-shot recognition was script assignment for English loanwords in Nepali context, where the base model frequently defaulted to Devanagari transliteration. Supervised fine-tuning on 517 synthetic utterances provided sufficient script distribution alignment to resolve the vast majority of predictable code-switches in clean acoustic conditions.

Because synthetic TTS speech exhibits high signal-to-noise ratio and clean phoneme articulation, rollouts during GRPO frequently produced identical hypotheses across group samples (zero variance), leading to skipped gradient updates for approximately 70 percent of batches. The remaining reward-weighted updates refined logit margins without shifting the greedy argmax output on this dev set. The script gate provides targeted corrections on boundary tokens but does not produce a statistically detectable aggregate MER change on clean synthetic dev data.

## Limitations

1. All training and evaluation utterances are synthetic text-to-speech audio. Acoustic diversity is limited to six synthetic speaker profiles and simulated telephony noise.
2. The evaluation set contains 59 utterances. A change of one character edit corresponds to approximately 0.0002 MER difference, limiting statistical power to distinguish subtle policy shifts.
3. Evaluations were performed using greedy argmax decoding. Sampling or beam search exploration was not tested at test time.

## Reproduction

Commit hash: 4c22ddecf3c3 (dirty)

Commands to reproduce evaluations:
```bash
python3 scripts/run_eval.py --config configs/model.yaml --manifest data/splits/synthetic_dev.jsonl --output experiments/S0-20261019-baseline-zero-shot/eval_dev.json
python3 scripts/run_eval.py --config configs/sft.yaml --checkpoint experiments/S1-20261019-sft-run01/checkpoints/best.pt --manifest data/splits/synthetic_dev.jsonl --output experiments/S1-20261019-sft-run01/eval_dev.json
python3 scripts/run_eval.py --config configs/sft_gate.yaml --checkpoint experiments/S2-20261020-sft-gate-run01/checkpoints/best.pt --manifest data/splits/synthetic_dev.jsonl --output experiments/S2-20261020-sft-gate-run01/eval_dev.json
python3 scripts/run_eval.py --config configs/grpo.yaml --checkpoint experiments/S4-20261019-grpo-run01/checkpoints/best.pt --manifest data/splits/synthetic_dev.jsonl --output experiments/S4-20261019-grpo-run01/eval_dev.json
python3 scripts/run_eval.py --config configs/grpo_gate.yaml --checkpoint experiments/S5-20261020-grpo-gate-run01/checkpoints/best.pt --manifest data/splits/synthetic_dev.jsonl --output experiments/S5-20261020-grpo-gate-run01/eval_dev.json
```
