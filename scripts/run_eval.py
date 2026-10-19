"""Evaluate a system on one split. Test evaluation verifies the frozen test hash first."""

from pathlib import Path

import _cli

from csrl.adapters.factory import build_policy
from csrl.adapters.indic_transcribe import OfficialTranscriber, model_settings
from csrl.application.checkpoint import load_checkpoint
from csrl.application.evaluate import score_pairs, summarize
from csrl.application.settings import common_settings
from csrl.application.training import RunPaths, evaluate_policy, write_evaluation
from csrl.data.manifest import load_manifest, verify_frozen
from csrl.domain import constants
from csrl.domain.enums import MetricName, Split
from csrl.domain.errors import CsrlError
from csrl.infrastructure.config import load_yaml

_OFFICIAL = "official"
_CHECKPOINT = "checkpoint"


def main() -> None:
    parser = _cli.run_parser(__doc__)
    parser.add_argument("--split", choices=[s.value for s in Split], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-base", type=Path, required=True)
    parser.add_argument("--decoder", choices=[_OFFICIAL, _CHECKPOINT], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-config", type=Path,
                        help="SFT or GRPO config the checkpoint was trained with")
    arguments = parser.parse_args()
    split = Split(arguments.split)
    if split is Split.TRAIN:
        raise CsrlError("evaluation on the train split is not a reported result")
    if split is Split.TEST:
        verify_frozen(arguments.manifest,
                      arguments.manifest.parent / constants.TEST_HASH_FILE_NAME)
    config_files = [arguments.model_config]
    if arguments.decoder == _CHECKPOINT:
        if arguments.checkpoint is None or arguments.train_config is None:
            raise CsrlError("--decoder checkpoint needs --checkpoint and --train-config")
        config_files.append(arguments.train_config)
    directory, commit = _cli.start_run(arguments, config_files)
    try:
        utterances = load_manifest(arguments.manifest)
        model_config = load_yaml(arguments.model_config)
        if arguments.decoder == _OFFICIAL:
            transcriber = OfficialTranscriber(
                model_settings(model_config, require_prompt=False), arguments.audio_base)
            scores = score_pairs(utterances, transcriber.transcribe(utterances))
            summary = summarize(scores)
        else:
            common = common_settings(load_yaml(arguments.train_config))
            policy = build_policy(model_settings(model_config, require_prompt=True), common)
            load_checkpoint(arguments.checkpoint, policy.backbone, policy.gate)
            summary, scores = evaluate_policy(
                policy, utterances, RunPaths(directory, arguments.audio_base), common.batch_size)
        write_evaluation(directory, split, summary, scores)
    except Exception as error:
        _cli.record(arguments, commit, config_files, _cli.STATUS_INVALID, _cli.EMPTY_CELL,
                    _cli.EMPTY_CELL, f"{type(error).__name__}: {error}")
        raise
    status = _cli.STATUS_TEST if split is Split.TEST else _cli.STATUS_DONE
    _cli.record(arguments, commit, config_files, status,
                f"{summary[MetricName.MER.value]:.4f}", f"{summary[MetricName.SCR.value]:.4f}",
                f"split={split.value} manifest={arguments.manifest}")


if __name__ == "__main__":
    main()
