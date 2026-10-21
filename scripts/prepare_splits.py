"""Build synthetic train and dev, real dev and test, check text leakage, freeze real test."""

import argparse
import logging
from pathlib import Path

from csrl.application.synthesis import sentence_id_of
from csrl.data.leakage import check_leakage
from csrl.data.manifest import file_sha256, load_manifest, write_manifest
from csrl.data.splits import split_by_speaker
from csrl.domain import constants
from csrl.domain.enums import Split
from csrl.domain.errors import LeakageError, SplitError
from csrl.infrastructure.config import load_yaml, require
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)
SYNTHETIC_PREFIX = "synthetic_"
REAL_PREFIX = "real_"


def _ratios(section: dict[str, object]) -> dict[Split, float]:
    return {Split(name): float(value) for name, value in section.items()}  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="split only synthetic data into train and dev")
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    config = load_yaml(arguments.config)
    seed = int(require(config, "seed"))
    splits_directory = Path(str(require(config, "splits_dir")))

    synthetic = []
    for manifest in require(config, "synthetic_manifests"):
        synthetic.extend(load_manifest(Path(str(manifest))))

    synthetic_split = split_by_speaker(synthetic, _ratios(require(config, "synthetic_split_ratios")),
                                       seed, group_key=sentence_id_of)
    for split, utterances in synthetic_split.utterances.items():
        path = splits_directory / f"{SYNTHETIC_PREFIX}{split.value}{constants.MANIFEST_SUFFIX}"
        write_manifest(utterances, path)
        hours = sum(u.duration_seconds for u in utterances) / constants.SECONDS_PER_HOUR
        _logger.info("%s: %d utterances, %d groups, %.2f h", path, len(utterances),
                     len(synthetic_split.speakers[split]), hours)
        (splits_directory / f"{SYNTHETIC_PREFIX}{split.value}{constants.SPEAKER_LIST_SUFFIX}").write_text(
            "\n".join(sorted(synthetic_split.speakers[split])) + "\n", encoding=constants.TEXT_ENCODING)

    if arguments.synthetic_only:
        _logger.info("synthetic-only mode: skipping real recordings split")
        return

    frozen = splits_directory / constants.TEST_HASH_FILE_NAME
    if frozen.exists():
        raise SplitError(f"{frozen} exists; the real test split is frozen")

    real = load_manifest(Path(str(require(config, "real_recordings_manifest"))))
    valid_real = [u for u in real if u.duration_seconds <= constants.MAX_CLIP_SECONDS]
    excluded_long = len(real) - len(valid_real)
    if excluded_long:
        _logger.info("excluded %d clips longer than %.1f s per data_spec.md", excluded_long, constants.MAX_CLIP_SECONDS)
    real = valid_real

    report = check_leakage([u.text for u in synthetic], [u.text for u in real],
                           int(require(config, "leakage_ngram_size")))
    if not report.is_clean:
        for text in report.exact_matches[:10]:
            _logger.error("exact match: %s", text)
        for text, shared in report.ngram_overlaps[:10]:
            _logger.error("shared n-gram %r in: %s", shared, text)
        raise LeakageError(f"{len(report.exact_matches)} exact and "
                           f"{len(report.ngram_overlaps)} n-gram overlaps with real speech")

    real_split = split_by_speaker(real, _ratios(require(config, "real_split_ratios")), seed)
    for split, utterances in real_split.utterances.items():
        path = splits_directory / f"{REAL_PREFIX}{split.value}{constants.MANIFEST_SUFFIX}"
        write_manifest(utterances, path)
        hours = sum(u.duration_seconds for u in utterances) / constants.SECONDS_PER_HOUR
        _logger.info("%s: %d utterances, %d groups, %.2f h", path, len(utterances),
                     len(real_split.speakers[split]), hours)
        (splits_directory / f"{REAL_PREFIX}{split.value}{constants.SPEAKER_LIST_SUFFIX}").write_text(
            "\n".join(sorted(real_split.speakers[split])) + "\n", encoding=constants.TEXT_ENCODING)
    test_path = splits_directory / f"{REAL_PREFIX}{Split.TEST.value}{constants.MANIFEST_SUFFIX}"
    frozen.write_text(file_sha256(test_path) + "\n", encoding=constants.TEXT_ENCODING)
    _logger.info("froze %s", test_path)


if __name__ == "__main__":
    main()
