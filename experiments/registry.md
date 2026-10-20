# Experiment Registry

Append only. Never edit or delete a row. Status values: RUNNING, DONE, INVALID, TEST.

| run_id | date | system | commit | config | status | dev MER | dev SCR | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0-20261019-baseline-zero-shot | 2026-10-19 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261019-baseline-zero-shot | 2026-10-19 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | DONE | 0.0590 | 0.0232 | split=dev manifest=data/splits/synthetic_dev.jsonl |
| S1-20261019-sft-baseline | 2026-10-19 | S1 | 4c22ddecf3c3 (dirty) | sft.yaml, model.yaml | RUNNING | - | - | - |
| S1-20261019-sft-baseline | 2026-10-19 | S1 | 4c22ddecf3c3 (dirty) | sft.yaml, model.yaml | INVALID | - | - | interrupted by user before training started |
| S1-20261019-sft-run01 | 2026-10-19 | S1 | 4c22ddecf3c3 (dirty) | sft.yaml, model.yaml | RUNNING | - | - | - |
| S1-20261019-sft-run01 | 2026-10-19 | S1 | 4c22ddecf3c3 (dirty) | sft.yaml, model.yaml | DONE | 0.0200 | - | best dev MER; SCR in eval_dev.json |
| S4-20261019-grpo-run01 | 2026-10-19 | S4 | 4c22ddecf3c3 (dirty) | grpo.yaml, model.yaml | RUNNING | - | - | - |
| S4-20261019-grpo-run01 | 2026-10-19 | S4 | 4c22ddecf3c3 (dirty) | grpo.yaml, model.yaml | DONE | 0.0200 | - | best dev MER; SCR in eval_dev.json |
| S2-20261020-sft-gate-run01 | 2026-10-19 | S2 | 4c22ddecf3c3 (dirty) | sft_gate.yaml, model.yaml | RUNNING | - | - | - |
| S2-20261020-sft-gate-run01 | 2026-10-19 | S2 | 4c22ddecf3c3 (dirty) | sft_gate.yaml, model.yaml | DONE | 0.0202 | - | best dev MER; SCR in eval_dev.json |
| S5-20261020-grpo-gate-run01 | 2026-10-19 | S5 | 4c22ddecf3c3 (dirty) | grpo_gate.yaml, model.yaml | RUNNING | - | - | - |
| S5-20261020-grpo-gate-run01 | 2026-10-19 | S5 | 4c22ddecf3c3 (dirty) | grpo_gate.yaml, model.yaml | DONE | 0.0202 | - | best dev MER; SCR in eval_dev.json |
| S0-20261020-real-dev-baseline | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261020-real-dev-baseline | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | INVALID | - | - | ValueError: audio is 1000s; this wrapper decodes whole files and the checkpoint degrades past ~45s (it trains at max_duration 30). For long audio use the chunked helper that ships in this repository:
    from long_form import transcribe_long
    text = transcribe_long(asr, audio, lang='hi')
Or raise max_seconds to accept a truncated transcript. |
| S0-20261020-real-dev-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261020-real-dev-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261020-real-dev-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | DONE | 0.2136 | 0.0370 | split=dev manifest=data/splits/real_dev.jsonl |
| S1-20261020-sft-run02 | 2026-10-20 | S1 | 4c22ddecf3c3 (dirty) | sft.yaml, model.yaml | RUNNING | - | - | - |
| S1-20261020-sft-run02 | 2026-10-20 | S1 | 4c22ddecf3c3 (dirty) | sft.yaml, model.yaml | DONE | 0.0200 | - | best dev MER; SCR in eval_dev.json |
| S1-20261020-real-dev-run01 | 2026-10-20 | S1 | 4c22ddecf3c3 (dirty) | model.yaml, sft.yaml | RUNNING | - | - | - |
| S1-20261020-real-dev-run01 | 2026-10-20 | S1 | 4c22ddecf3c3 (dirty) | model.yaml, sft.yaml | DONE | 0.1982 | 0.0325 | split=dev manifest=data/splits/real_dev.jsonl |
| S2-20261020-sft-gate-run02 | 2026-10-20 | S2 | 4c22ddecf3c3 (dirty) | sft_gate.yaml, model.yaml | RUNNING | - | - | - |
| S2-20261020-sft-gate-run02 | 2026-10-20 | S2 | 4c22ddecf3c3 (dirty) | sft_gate.yaml, model.yaml | DONE | 0.0193 | - | best dev MER; SCR in eval_dev.json |
| S2-20261020-real-dev-run01 | 2026-10-20 | S2 | 4c22ddecf3c3 (dirty) | model.yaml, sft_gate.yaml | RUNNING | - | - | - |
| S2-20261020-real-dev-run01 | 2026-10-20 | S2 | 4c22ddecf3c3 (dirty) | model.yaml, sft_gate.yaml | DONE | 0.2004 | 0.0361 | split=dev manifest=data/splits/real_dev.jsonl |
| S4-20261020-grpo-run02 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | grpo.yaml, model.yaml | RUNNING | - | - | - |
| S4-20261020-grpo-run02 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | grpo.yaml, model.yaml | DONE | 0.0200 | - | best dev MER; SCR in eval_dev.json |
| S4-20261020-real-dev-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | RUNNING | - | - | - |
| S4-20261020-real-dev-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | DONE | 0.1980 | 0.0324 | split=dev manifest=data/splits/real_dev.jsonl |
| S0-20261020-bhasaanuvad-reg-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261020-bhasaanuvad-reg-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | DONE | 0.2080 | 0.0871 | split=dev manifest=data/splits/bhasaanuvad_eval.jsonl |
| S4-20261020-bhasaanuvad-reg-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | RUNNING | - | - | - |
| S4-20261020-bhasaanuvad-reg-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | DONE | 0.1842 | 0.0695 | split=dev manifest=data/splits/bhasaanuvad_eval.jsonl |
| S0-20261020-librispeech-reg-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261020-librispeech-reg-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | DONE | 0.0421 | 0.0059 | split=dev manifest=data/splits/librispeech_eval.jsonl |
| S4-20261020-librispeech-reg-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | RUNNING | - | - | - |
| S4-20261020-librispeech-reg-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | DONE | 0.0527 | 0.0103 | split=dev manifest=data/splits/librispeech_eval.jsonl |
| S0-20261020-real-test-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | RUNNING | - | - | - |
| S0-20261020-real-test-run01 | 2026-10-20 | S0 | 4c22ddecf3c3 (dirty) | model.yaml | TEST | 0.2696 | 0.0394 | split=test manifest=data/splits/real_test.jsonl |
| S1-20261020-real-test-run01 | 2026-10-20 | S1 | 4c22ddecf3c3 (dirty) | model.yaml, sft.yaml | RUNNING | - | - | - |
| S1-20261020-real-test-run01 | 2026-10-20 | S1 | 4c22ddecf3c3 (dirty) | model.yaml, sft.yaml | TEST | 0.2460 | 0.0362 | split=test manifest=data/splits/real_test.jsonl |
| S4-20261020-real-test-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | RUNNING | - | - | - |
| S4-20261020-real-test-run01 | 2026-10-20 | S4 | 4c22ddecf3c3 (dirty) | model.yaml, grpo.yaml | TEST | 0.2457 | 0.0360 | split=test manifest=data/splits/real_test.jsonl |
