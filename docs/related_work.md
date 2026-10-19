# Related Work and Novelty

Checked on 2026-10-19. Every agent must reread this file before writing any claim of novelty. If a newer paper overlaps with this project, add it here before continuing.

## 1. Closest prior work

**Ye and Vickers, Reinforcement Learning for Data-Efficient Code-Switched ASR, arXiv 2607.02757 (July 2026).**

- Uses GRPO with an error-rate reward plus a script-fidelity reward that penalizes output in the wrong writing system.
- Adds a two-pass draft-and-refinement procedure.
- Base model: Qwen2-Audio, an audio-language model.
- Covers 10 language pairs and trains on TTS-generated code-switched speech only.
- Reports that RLVR matches LoRA SFT with a fraction of the data, and that the script-fidelity reward reduces script contamination.
- Link: https://arxiv.org/abs/2607.02757

Consequence: "GRPO with a script-aware reward for code-switched ASR" is not new by itself. This project must not claim it as a contribution, and must cite this paper.

**Reducing Language Confusion for Code-switching Speech Recognition with Token-level Language Diarization, arXiv 2210.14567.** Authors not verified in this check; read them from the paper before citing.

- Adds an auxiliary decoder that predicts a language label for every output token, and biases the ASR decoder with these posteriors.
- Mandarin-English, SEAME corpus. No reinforcement learning.
- Link: https://arxiv.org/abs/2210.14567

Consequence: a token-level script or language predictor is not new by itself. Our gate is a lighter variant (a linear head on the decoder state, with an optional logit bias).

**Kumar et al., When Synthetic Speech Is All You Have: Better Call GRPO, arXiv 2607.08409 (July 2026).** GRPO beats SFT when adapting an LLM-based ASR to a domain using synthetic TTS speech only; the authors attribute the gain to fewer insertions and better audio anchoring. Not code-switching. Link: https://arxiv.org/abs/2607.08409

**Background:** RL for ASR in general is established (arXiv 2509.01939, 2509.18569, 2607.08409, 2609.14455; see `docs/references.md`). Hindi-English and Bengali-English code-switched ASR benchmarks exist (MUCS 2021).

## 2. What remains open, and what this project claims

Each claim below is a candidate contribution. A claim counts only if the experiments support it.

1. **Encoder-decoder ASR, not an audio-language model.** Prior RL code-switching work uses Qwen2-Audio. We test whether the same family of rewards helps a dedicated FastConformer encoder-decoder ASR model that is already code-mix aware (indic-transcribe-flex). This tests whether the gains survive a much stronger, task-specific starting point.
2. **Synthetic-to-real transfer, measured.** Training is synthetic, as in prior work, but every claim is measured on a new real Nepali-English test set, and the synthetic data design is ablated (voices, TTS systems, expressive style, telephony augmentation). Prior work reports that RL works on synthetic speech; this asks which synthetic properties transfer for code-switching. As of this check we found no published Nepali-English code-switched ASR benchmark. Verify this again before claiming it.
3. **Explicit structure plus RL.** The prior token-level work uses no RL, and the prior RL work uses no token-level module. The 2 by 2 design (gate on or off, script reward on or off) tests whether they are complementary or redundant. Either answer is a result.
4. **Where errors happen.** Switch-point error rate (SPER) and per-direction script confusion (SCR_ne, SCR_en) give a finer analysis than overall error rate plus a single contamination score.
5. **Objective comparison for ASR.** A controlled comparison of GRPO, DAPO, Dr. GRPO, and GSPO at equal sample budget on a speech task. We found no such comparison for ASR. Verify this again before claiming it.

## 3. Honest assessment

- The project is publishable at workshop or Interspeech-short level if claims 1 to 3 produce clear results with confidence intervals.
- The strongest single contribution is likely claim 2, combined with releasing the evaluation set if data permissions allow.
- Risk: indic-transcribe-flex may already make few script errors on Nepali. If S0 has little headroom (see the stopping rule in `docs/research_plan.md`), the project becomes a negative or analysis result. That is still reportable, but weaker.
- Required in every report: cite arXiv 2607.02757 and arXiv 2210.14567, and state clearly how this work differs.
