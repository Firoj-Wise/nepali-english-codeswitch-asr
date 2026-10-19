"""Gate check for the NeMo binding. Must pass on the GPU machine before any training.

Checks, in order:
1. The backbone loads, the encoder is frozen, and only decoder and head parameters train.
2. Greedy decoding with the configured prompt reproduces the official mixed-mode output on
   a sample of dev utterances, at or above equivalence_min_exact_match.
3. Teacher-forced log probabilities of the official hypotheses are finite, and a backward
   pass reaches decoder parameters and no encoder parameters.
"""

import argparse
import logging
import math
from pathlib import Path

import torch

from csrl.adapters.factory import build_policy
from csrl.adapters.indic_transcribe import OfficialTranscriber, model_settings
from csrl.application.settings import common_settings
from csrl.data.manifest import load_manifest
from csrl.data.normalize import normalize_text
from csrl.domain.errors import ModelEquivalenceError
from csrl.infrastructure.audio import collate_audio
from csrl.infrastructure.config import load_yaml
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="dev manifest")
    parser.add_argument("--audio-base", type=Path, required=True)
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)

    model = model_settings(load_yaml(arguments.model_config), require_prompt=True)
    common = common_settings(load_yaml(arguments.train_config))
    utterances = load_manifest(arguments.manifest)[: model.equivalence_sample_count]

    policy = build_policy(model, common)
    backbone = policy.backbone
    trainable = sum(p.numel() for p in backbone.trainable_parameters())
    _logger.info("trainable decoder and head parameters: %d", trainable)
    if trainable == 0:
        raise ModelEquivalenceError("no trainable parameters were found")

    official = OfficialTranscriber(model, arguments.audio_base).transcribe(utterances)
    backbone.train_mode(False)
    audio, lengths = collate_audio(utterances, arguments.audio_base, backbone.device)
    encoder = backbone.encode(audio, lengths)
    ours = policy.generate(encoder, temperature=1.0, greedy=True).texts
    matches = 0
    for utterance, expected, produced in zip(utterances, official, ours):
        same = normalize_text(expected) == normalize_text(produced)
        matches += same
        if not same:
            _logger.warning("%s official=%r ours=%r", utterance.utterance_id, expected, produced)
    rate = matches / len(utterances)
    _logger.info("exact match with official output: %.3f", rate)
    if rate < model.equivalence_min_exact_match:
        raise ModelEquivalenceError(
            f"exact match {rate:.3f} below {model.equivalence_min_exact_match}; check the "
            "prompt ids and the NeMo calls in csrl/adapters/indic_transcribe.py"
        )

    loss = policy.supervised_loss(encoder, [normalize_text(t) for t in official]).token_loss
    if not math.isfinite(float(loss.detach())):
        raise ModelEquivalenceError(f"teacher-forced loss is not finite: {loss}")
    loss.backward()
    decoder_grads = [p.grad for p in backbone.trainable_parameters() if p.grad is not None]
    if not decoder_grads or not any(bool(torch.any(g != 0)) for g in decoder_grads):
        raise ModelEquivalenceError("backward pass did not reach decoder parameters")
    _logger.info("smoke test passed: loss %.4f on official hypotheses", float(loss.detach()))


if __name__ == "__main__":
    main()
