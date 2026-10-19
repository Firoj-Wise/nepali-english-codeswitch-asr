"""Check sentences planned for synthesis against sentences planned for human recording.

Run this before synthesis and before recording, so leaking sentences are removed while it
is still cheap. prepare_splits.py repeats the check on the final manifests.
"""

import argparse
import logging
from pathlib import Path

from csrl.data.leakage import check_leakage
from csrl.data.sentences import load_sentences
from csrl.domain.errors import LeakageError
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)
_REPORT_LIMIT = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-sentences", type=Path, required=True)
    parser.add_argument("--recording-sentences", type=Path, required=True)
    parser.add_argument("--ngram-size", type=int, required=True)
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    report = check_leakage(
        [s.text for s in load_sentences(arguments.synthesis_sentences)],
        [s.text for s in load_sentences(arguments.recording_sentences)],
        arguments.ngram_size,
    )
    for text in report.exact_matches[:_REPORT_LIMIT]:
        _logger.error("exact match: %s", text)
    for text, shared in report.ngram_overlaps[:_REPORT_LIMIT]:
        _logger.error("shared n-gram %r in: %s", shared, text)
    if not report.is_clean:
        raise LeakageError(f"{len(report.exact_matches)} exact and "
                           f"{len(report.ngram_overlaps)} n-gram overlaps")
    _logger.info("no leakage found")


if __name__ == "__main__":
    main()
