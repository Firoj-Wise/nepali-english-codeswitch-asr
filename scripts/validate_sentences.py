"""Check generated code-switched sentences against the transcription convention."""

import argparse
import logging
from collections import Counter
from pathlib import Path

from _data_io import with_suffix_tag, write_rejections

from csrl.adapters.synthesis_factory import sentence_limits
from csrl.data.sentences import load_sentences, validate_sentences, write_sentences
from csrl.infrastructure.config import load_yaml, require
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    config = load_yaml(arguments.config)
    raw = Path(str(require(config, "raw_sentences")))
    output = Path(str(require(config, "validated_sentences")))
    result = validate_sentences(load_sentences(raw), sentence_limits(config))
    write_sentences(result.valid, output)
    write_rejections(with_suffix_tag(output, "rejected"), result.rejected)
    reasons = Counter(r.reason for r in result.rejected)
    _logger.info("kept %d sentences, rejected %d: %s", len(result.valid),
                 len(result.rejected), dict(reasons))
    _logger.info("wrote %s", output)


if __name__ == "__main__":
    main()
