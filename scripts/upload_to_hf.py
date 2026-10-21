#!/usr/bin/env python3
"""Upload verified clean podcast dataset and splits to Hugging Face Hub."""

import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)
CSRL_ROOT = Path(__file__).resolve().parent.parent
DATASTORE = Path("/home/oem/wiseyak_backup/wiseai-training-pipeline/datastore/nepali_english_codeswitched_podcasts")
SPLITS_DIR = CSRL_ROOT / "data" / "splits"

REPOS = [
    "SamaFiroz/nepali-english-codeswitch-podcasts",
    "Firoj112/nepali-english-codeswitch-podcasts"
]


def main() -> None:
    configure_logging(logging.INFO, None)
    token = os.environ.get("HF_TOKEN")
    if not token:
        _logger.error("HF_TOKEN environment variable not set.")
        sys.exit(1)

    api = HfApi(token=token)

    for repo_id in REPOS:
        _logger.info("Uploading clean dataset to %s...", repo_id)

        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        except Exception as e:
            _logger.warning("Repo check: %s", e)

        # 1. Upload README.md
        readme_path = DATASTORE / "README.md"
        if readme_path.exists():
            _logger.info("Uploading README.md to %s...", repo_id)
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset"
            )

        # 2. Upload manifest.jsonl
        manifest_path = DATASTORE / "manifest.jsonl"
        if manifest_path.exists():
            _logger.info("Uploading clean manifest.jsonl to %s...", repo_id)
            api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo="manifest.jsonl",
                repo_id=repo_id,
                repo_type="dataset"
            )

        # 3. Upload real_dev.jsonl and real_test.jsonl
        dev_path = SPLITS_DIR / "real_dev.jsonl"
        if dev_path.exists():
            _logger.info("Uploading real_dev.jsonl to %s...", repo_id)
            api.upload_file(
                path_or_fileobj=str(dev_path),
                path_in_repo="real_dev.jsonl",
                repo_id=repo_id,
                repo_type="dataset"
            )

        test_path = SPLITS_DIR / "real_test.jsonl"
        if test_path.exists():
            _logger.info("Uploading real_test.jsonl to %s...", repo_id)
            api.upload_file(
                path_or_fileobj=str(test_path),
                path_in_repo="real_test.jsonl",
                repo_id=repo_id,
                repo_type="dataset"
            )

        # 4. Upload clips folder
        clips_dir = DATASTORE / "clips"
        if clips_dir.exists():
            _logger.info("Uploading audio clips to %s (this may take a few minutes)...", repo_id)
            api.upload_folder(
                folder_path=str(clips_dir),
                path_in_repo="clips",
                repo_id=repo_id,
                repo_type="dataset"
            )

        _logger.info("Successfully uploaded dataset to %s!", repo_id)

    _logger.info("ALL REPOSITORIES UPDATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
