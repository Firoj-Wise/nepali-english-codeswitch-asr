# Data Specification

## 1. Overview

| Pool | Origin | Use |
| --- | --- | --- |
| Synthetic train | Generated sentences rendered by TTS | SFT and RL training |
| Synthetic dev | Same pipeline, held-out sentences | Training diagnostics only |
| Real dev | Human recordings, speaker-disjoint from real test | Model selection and tuning |
| Real test | Human recordings, frozen hash | Final numbers only |
| FLEURS ne_np test, LibriSpeech test-clean | Public | Monolingual regression checks |

Training data is synthetic. Every claim in the paper is measured on real speech.

## 2. Sources and licenses

| Name | Use | License | Link |
| --- | --- | --- | --- |
| Generated code-switched sentences | synthesis input | produced for this project | `docs/prompts/text_generation_prompt.md` |
| In-house fine-tuned OmniVoice (Wiseyak) | main TTS | internal; weights and features not released | not public |
| Gemini 3.8 Flash TTS | reproducible TTS arm | Gemini API terms | https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-tts |
| OmniVoice base model (k2-fsa) | reproducible TTS arm | Apache 2.0 | https://github.com/k2-fsa/OmniVoice |
| Real human recordings | real dev and test | consent per `docs/real_test_set_protocol.md` | not public unless consent allows |
| OpenSLR 54, Large Nepali ASR training set | optional SFT auxiliary | CC BY-SA 4.0 | https://openslr.org/54/ |
| Google FLEURS, Nepali (ne_np) | regression check | CC BY 4.0 | https://huggingface.co/datasets/google/fleurs |
| LibriSpeech test-clean | regression check | CC BY 4.0 | https://www.openslr.org/12 |

Re-check every license on its source page before use, and record the date checked here.

## 3. Synthetic pipeline

1. **Text.** Generate sentences with `docs/prompts/text_generation_prompt.md`. Validate them with `scripts/validate_sentences.py`, which applies the transcription convention, requires at least one Devanagari and one Latin word, enforces length limits, and removes duplicates.
2. **Leakage pre-check.** Run `scripts/check_leakage.py` between the synthesis pool and the reading set.
3. **Synthesis.** Run `scripts/synthesize.py --system <system> --style <style>`. Each sentence is rendered by `voices_per_sentence` voices, chosen deterministically. The run is resumable, and failures are written to `failures.jsonl`.
4. **Round-trip filter.** Run `scripts/filter_synthetic.py` with an independent Nepali ASR, which must not be indic-transcribe-flex. The default scope is `devanagari_only`, which scores the Nepali words and checks the speaking rate (see the module docstring for the exact rule).
5. **Optional telephony arm.** Run `scripts/augment_telephony.py`, which applies an 8 kHz band limit, mu-law companding, and noise at a random SNR. The output is deterministic per utterance.
6. **Splits.** Run `scripts/prepare_splits.py`. It splits synthetic data by sentence and real data by speaker, rejects any text leakage, and freezes the real test hash.

Confidentiality: the in-house model is used only through `src/csrl/adapters/omnivoice_tts.py` with a path outside the repository. Confidential features such as style tags are applied by an external style plugin. No company code, weights, configs, or tag lists enter this repository.

## 4. Manifest format

JSON Lines, one utterance per line, with fields named in `enums.ManifestField`:

- `utterance_id` (string, unique). For synthetic data: `<sentence_id>__<system>-<voice>-<style>`.
- `audio_path` (string, relative to `audio_base_directory` or `manifest_base_directory`)
- `duration_seconds` (float)
- `speaker_id` (string). For synthetic data: `<system>:<voice>:<style>`.
- `text` (string, in the convention of section 5)
- `source` (value of `enums.DataSource`)

Missing fields raise `ManifestSchemaError`. No defaults are filled in.

## 5. Transcription convention

- Nepali words in Devanagari. English words in Latin script, as spoken, not translated.
- A Nepali suffix or postposition after an English word is a separate word ("phone मा"). Normalization enforces this split in any case (decision 0002).
- Numbers as words, in the language spoken.
- Real utterances that break the convention are listed in `data/audit/convention_violations.jsonl` and excluded from real test, not corrected silently.

## 6. Audio constraints

- 16 kHz mono everywhere. TTS output is resampled at synthesis time; nothing is resampled at load time.
- Clips longer than `constants.MAX_CLIP_SECONDS` are excluded and counted. The base model refuses single-pass audio longer than 45 seconds.

## 7. Data report (a paper deliverable)

Report the following:

- **Sentence pipeline:** generated, validated, and rejected counts by reason; the native-speaker naturalness spot check.
- **Synthesis:** hours per TTS system, voice, and style; synthesis failures; filter rejections by reason.
- **Real data:** real dev and real test hours, speakers, read versus spontaneous share, and inter-annotator agreement.
- **Leakage:** the leakage check result.
