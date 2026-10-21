"""Round-trip filter a synthetic manifest with an independent ASR."""

import argparse
import logging
from collections import Counter
from pathlib import Path

from _data_io import with_suffix_tag, write_rejections

from csrl.adapters.hf_transcriber import HuggingFaceTranscriber
from csrl.adapters.synthesis_factory import filter_settings
from csrl.application.quality_filter import filter_synthetic
from csrl.data.manifest import load_manifest, write_manifest
from csrl.infrastructure.config import load_yaml, require
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    config = load_yaml(arguments.config)
    section = require(config, "filter")
    transcriber = HuggingFaceTranscriber(
        str(require(section, "model_id")), str(require(section, "device")),
        int(require(section, "batch_size")), Path(str(require(config, "audio_base_directory"))),
    )
    result = filter_synthetic(load_manifest(arguments.manifest), transcriber,
                              filter_settings(config))
    write_manifest(result.kept, with_suffix_tag(arguments.manifest, "filtered"))
    write_rejections(with_suffix_tag(arguments.manifest, "rejected"), result.rejected)
    _logger.info("rejections by reason: %s", dict(Counter(r.reason for r in result.rejected)))


if __name__ == "__main__":
    main()
