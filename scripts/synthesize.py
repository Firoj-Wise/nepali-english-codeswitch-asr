"""Render validated sentences to 16 kHz audio with one TTS system and one delivery style.

Resumable: items already in the output manifest are skipped. Run again after a crash.
"""

import argparse
import logging
from pathlib import Path

from csrl.adapters.synthesis_factory import build_tts, synthesis_settings, voice_pool
from csrl.application.synthesis import plan_assignments, run_synthesis
from csrl.data.sentences import load_sentences
from csrl.domain.enums import DeliveryStyle, TtsSystem
from csrl.infrastructure.config import load_yaml, require
from csrl.infrastructure.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--system", choices=[s.value for s in TtsSystem], required=True)
    parser.add_argument("--style", choices=[s.value for s in DeliveryStyle], required=True)
    parser.add_argument("--limit", type=int, help="pilot runs: only the first N sentences")
    arguments = parser.parse_args()
    config = load_yaml(arguments.config)
    system = TtsSystem(arguments.system)
    style = DeliveryStyle(arguments.style)
    base = Path(str(require(config, "audio_base_directory")))
    name = f"synth_{system.value}_{style.value}"
    output_directory = base / "synthetic" / name
    configure_logging(logging.INFO, output_directory / "synthesis.log")
    sentences = load_sentences(Path(str(require(config, "validated_sentences"))))
    if arguments.limit is not None:
        sentences = sentences[: arguments.limit]
    settings = synthesis_settings(config)
    plan = plan_assignments(sentences, voice_pool(config, system, style), settings)
    run_synthesis(build_tts(config, system), plan, output_directory, base,
                  base / "manifests" / f"{name}.jsonl", settings)


if __name__ == "__main__":
    main()
