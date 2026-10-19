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
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    config = load_yaml(arguments.config)
    seed = int(require(config, "seed"))
    splits_directory = Path(str(require(config, "splits_dir")))
    frozen = splits_directory / constants.TEST_HASH_FILE_NAME
    if frozen.exists():
        raise SplitError(f"{frozen} exists; the real test split is frozen")

    synthetic = []
    for manifest in require(config, "synthetic_manifests"):
        synthetic.extend(load_manifest(Path(str(manifest))))
    real = load_manifest(Path(str(require(config, "real_recordings_manifest"))))

    report = check_leakage([u.text for u in synthetic], [u.text for u in real],
                           int(require(config, "leakage_ngram_size")))
    if not report.is_clean:
        for text in report.exact_matches[:10]:
            _logger.error("exact match: %s", text)
        for text, shared in report.ngram_overlaps[:10]:
            _logger.error("shared n-gram %r in: %s", shared, text)
        raise LeakageError(f"{len(report.exact_matches)} exact and "
                           f"{len(report.ngram_overlaps)} n-gram overlaps with real speech")

    synthetic_split = split_by_speaker(synthetic, _ratios(require(config, "synthetic_split_ratios")),
                                       seed, group_key=sentence_id_of)
    real_split = split_by_speaker(real, _ratios(require(config, "real_split_ratios")), seed)
    for prefix, result in ((SYNTHETIC_PREFIX, synthetic_split), (REAL_PREFIX, real_split)):
        for split, utterances in result.utterances.items():
            path = splits_directory / f"{prefix}{split.value}{constants.MANIFEST_SUFFIX}"
            write_manifest(utterances, path)
            hours = sum(u.duration_seconds for u in utterances) / constants.SECONDS_PER_HOUR
            _logger.info("%s: %d utterances, %d groups, %.2f h", path, len(utterances),
                         len(result.speakers[split]), hours)
            (splits_directory / f"{prefix}{split.value}{constants.SPEAKER_LIST_SUFFIX}").write_text(
                "\n".join(sorted(result.speakers[split])) + "\n", encoding=constants.TEXT_ENCODING)
    test_path = splits_directory / f"{REAL_PREFIX}{Split.TEST.value}{constants.MANIFEST_SUFFIX}"
    frozen.write_text(file_sha256(test_path) + "\n", encoding=constants.TEXT_ENCODING)
    _logger.info("froze %s", test_path)


if __name__ == "__main__":
    main()
