#!/usr/bin/env python3
"""Format and push the verified Nepali-English code-switched podcast dataset to Hugging Face Hub
with native interactive audio playback widgets and clean single-text schema.
"""

import json
import logging
import os
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import HfApi

CSRL_ROOT = Path(__file__).resolve().parent.parent
DATASTORE = Path("/home/oem/wiseyak_backup/wiseai-training-pipeline/datastore/nepali_english_codeswitched_podcasts")
CLIPS_DIR = DATASTORE / "clips"
SPLITS_DIR = CSRL_ROOT / "data" / "splits"

REPOS_CONFIG = [
    {
        "repo_id": "Firoj112/nepali-english-codeswitch-podcasts",
        "token": os.environ.get("HF_TOKEN")
    },
    {
        "repo_id": "SamaFiroz/nepali-english-codeswitch-podcasts",
        "token": os.environ.get("HF_TOKEN_SAMAFIRUZ") or os.environ.get("HF_TOKEN")
    }
]

DATASET_CARD = """---
task_categories:
  - automatic-speech-recognition
language:
  - ne
  - en
tags:
  - audio
  - speech
  - code-switching
  - nepali
  - english
pretty_name: Nepali-English Code-Switched Podcast Evaluation Benchmark
size_categories:
  - 10K<n<100K
---

# Nepali-English Code-Switched Podcast Evaluation Benchmark

This dataset is a gold-standard evaluation benchmark for Nepali-English code-switched Automatic Speech Recognition (ASR). It features 13,044 natural conversational speech clips (15.45 hours) across 28 distinct long-form podcast episodes recorded in Nepal.

## Dataset Structure

- **Validation Split (`validation`)**: 2,830 clips (3.55 hours)
- **Test Split (`test`)**: 10,214 clips (11.90 hours)

### Schema

- `utterance_id` (`string`): Unique identifier for the audio segment (e.g. `jpNejIBVYTM_0068`)
- `audio` (`audio`): 16 kHz mono WAV audio file with interactive browser playback
- `text` (`string`): Verbatim code-switched ground truth transcription (Devanagari for Nepali words, Latin script for English words)
- `duration_seconds` (`float64`): Clip length in seconds
- `speaker_id` (`string`): Diarized speaker identifier
- `source` (`string`): Source attribution (`real_human`)

## Transcription Guidelines

1. **Verbatim Fidelity**: All speech transcribed exactly as spoken. Natural conversational loops and fillers (`अँ`, `हैन`, `yeah yeah`, `so so`) are preserved.
2. **Strict Script Orthography**: Spoken Nepali matrix vocabulary is transcribed in Devanagari; English loanwords and technical vocabulary are transcribed in standard Latin English script (no phonetic transliterations).
3. **Detached Postpositions**: English loanwords with attached Nepali postpositions are separated with a space (e.g., `model हरू`, `phone मा`).
4. **Verbalized Numbers**: Spoken numbers are written as full words.
"""

def load_split_records(jsonl_path: Path) -> dict:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    data = {
        "utterance_id": [],
        "audio": [],
        "text": [],
        "duration_seconds": [],
        "speaker_id": [],
        "source": []
    }

    for r in records:
        clip_name = Path(r["audio_path"]).name
        local_clip = CLIPS_DIR / clip_name
        if not local_clip.exists():
            continue

        data["utterance_id"].append(r["utterance_id"])
        data["audio"].append(str(local_clip))
        data["text"].append(r.get("clean_verbatim_text") or r.get("text", ""))
        data["duration_seconds"].append(float(r.get("duration_seconds", 0.0)))
        data["speaker_id"].append(r.get("speaker_id", "unknown"))
        data["source"].append("real_human")

    return data


from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def main() -> None:
    configure_logging(logging.INFO, None)
    _logger.info("Loading validation records...")
    dev_data = load_split_records(SPLITS_DIR / "real_dev.jsonl")
    _logger.info("Validation: %d records.", len(dev_data['utterance_id']))

    _logger.info("Loading test records...")
    test_data = load_split_records(SPLITS_DIR / "real_test.jsonl")
    _logger.info("Test: %d records.", len(test_data['utterance_id']))

    _logger.info("Building Hugging Face Dataset with Audio feature...")
    dev_ds = Dataset.from_dict(dev_data).cast_column("audio", Audio(sampling_rate=16000))
    test_ds = Dataset.from_dict(test_data).cast_column("audio", Audio(sampling_rate=16000))

    ds_dict = DatasetDict({
        "validation": dev_ds,
        "test": test_ds
    })
    _logger.info("DatasetDict built: %s", ds_dict)

    for cfg in REPOS_CONFIG:
        repo_id = cfg["repo_id"]
        token = cfg["token"]
        api = HfApi(token=token)

        _logger.info("Configuring and pushing to %s...", repo_id)

        # Clean up any stale parquet files in repo to prevent schema conflicts
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            for f in files:
                if f.startswith("data/") and f.endswith(".parquet"):
                    _logger.info("Deleting stale parquet %s from %s...", f, repo_id)
                    api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            _logger.warning("Note on stale cleanup: %s", e)

        # Push clean dataset with embedded audio parquet shards
        _logger.info("Pushing DatasetDict to %s (this embeds audio for web player)...", repo_id)
        ds_dict.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=False,
            max_shard_size="500MB"
        )

        # Upload rich README.md with complete YAML frontmatter
        _logger.info("Updating README.md on %s...", repo_id)
        api.upload_file(
            path_or_fileobj=DATASET_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )

        _logger.info("Successfully published playable dataset to %s!", repo_id)

    _logger.info("ALL REPOSITORIES ARE NOW PLAYABLE AND CLEAN!")

if __name__ == "__main__":
    main()
