# Research Plan

## 0. Paper framing

Training data is synthetic, and evaluation is on real human speech. The question is which properties of synthetic code-switched speech transfer to real speakers, and whether RL closes the synthetic-to-real gap where SFT cannot. Two prior papers showed that RL helps on synthetic speech (arXiv 2607.08409 and arXiv 2607.02757). This project asks when and why it helps, for code-switching, on a strong encoder-decoder ASR model.

## 0.1 Data arms

| Arm | Training audio |
| --- | --- |
| D1 | In-house OmniVoice, neutral style |
| D2 | In-house OmniVoice, neutral plus expressive style |
| D3 | Public TTS only: Gemini TTS plus base OmniVoice (reproducible arm) |
| D4 | D2 plus D3: two or more TTS systems |
| D5 | D4 plus the telephony-augmented copy |

Every arm is matched in total hours. Compare arms with SFT first (system S1). Then run RL (S4) on the best arm and on D3, so a reproducible RL result exists.

## 1. Claim to test

Positioning against prior work is in `docs/related_work.md`. Read it first.

Script-aware sequence-level RL, optionally combined with a token-level script gate, reduces script confusion in Nepali-English code-switched ASR beyond supervised fine-tuning, without degrading monolingual Nepali or English accuracy.

## 2. Systems

| Id | System | Trained parts |
| --- | --- | --- |
| S0 | indic-transcribe-flex, zero-shot, mixed mode, lang ne | none |
| S1 | S0 plus SFT on train split | decoder |
| S2 | S1 plus script gate with auxiliary loss | decoder, gate |
| S3 | S1 plus GRPO with R_mer | decoder |
| S4 | S1 plus GRPO with R_mer plus R_scr | decoder |
| S5 | S2 plus GRPO with R_mer plus R_scr | decoder, gate |
| S6 | S5 plus R_hal | decoder, gate |

The core result is the 2 by 2 of gate (off, on) by script reward (off, on): S3, S4, S2 with R_mer only (optional), S5.

## 3. Evaluation

Every system is evaluated on: code-switched dev, code-switched test (final only), FLEURS ne_np test, LibriSpeech test-clean. Metrics per `docs/metrics_spec.md`.

## 4. Script gate design

A linear head on the final decoder hidden state predicting the script class of the next token (Devanagari, Latin, other). Training target is the script of the reference next token. Loss weight in `configs/sft.yaml`. At inference, an optional logit bias pushes down tokens whose script disagrees with a confident gate prediction. The bias strength and confidence threshold are tuned on dev only.

## 5. RL setup

The default objective is DAPO-style (asymmetric clipping, token-mean loss, dynamic sampling, no KL). GRPO, Dr. GRPO, and GSPO are available as config switches; see `docs/rl_objectives.md`. When a KL term is used, the reference model is the frozen init checkpoint. Only the decoder, the output head, and the gate are updated.

To run S2 or S5, set `script_gate.enabled: true` in the config. To run S3, set `reward.lambda_scr: 0.0`. To run S6, keep `reward.lambda_hal` positive; for S4 and S5 set it to 0.0.

## 6. Schedule (four days)

| Day | Work | Exit criterion |
| --- | --- | --- |
| 1 | Data audit, splits, metric tests, S0 evaluation | S0 numbers in registry |
| 2 | S1 and S2 training and evaluation | S1, S2 in registry |
| 3 | Smoke test, then S3 and S4, S5 if time | at least S3, S4 |
| 4 | Ablations, switch-point analysis, regression checks, report | report passes check |

## 7. Risks and stopping rules

- If S0 SCR on dev is already below `constants.SCR_HEADROOM_THRESHOLD`, stop and report that the headroom is too small before spending compute.
- If GRPO reward rises while dev MER rises, treat as reward hacking and inspect samples before continuing.
- If empty hypotheses rise during RL, halt the run.
