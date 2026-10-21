"""Create a telephony-augmented copy of a manifest (8 kHz band, mu-law, noise)."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from _data_io import with_suffix_tag

from csrl.adapters.synthesis_factory import telephony_settings
from csrl.data.augment import telephony
from csrl.data.manifest import load_manifest, write_manifest
from csrl.domain import constants
from csrl.infrastructure.audio import load_waveform, write_waveform
from csrl.infrastructure.config import load_yaml, require
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)
_TAG = "telephony"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    config = load_yaml(arguments.config)
    base = Path(str(require(config, "audio_base_directory")))
    settings, levels, subdirectory = telephony_settings(config)
    output = []
    for utterance in load_manifest(arguments.manifest):
        waveform = telephony(load_waveform(base / utterance.audio_path), utterance.utterance_id,
                             settings, levels)
        new_id = f"{utterance.utterance_id}{constants.SYNTHESIS_ID_SEPARATOR}{_TAG}"
        relative = Path(subdirectory) / f"{new_id}{constants.WAV_SUFFIX}"
        write_waveform(base / relative, waveform, constants.SAMPLE_RATE_HZ)
        output.append(replace(utterance, utterance_id=new_id, audio_path=str(relative)))
    target = with_suffix_tag(arguments.manifest, _TAG)
    write_manifest(output, target)
    _logger.info("wrote %d augmented utterances to %s", len(output), target)


if __name__ == "__main__":
    main()
