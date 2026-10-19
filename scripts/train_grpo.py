"""GRPO-family training from an SFT checkpoint (systems S3 to S6)."""

from pathlib import Path

import _cli

from csrl.adapters.factory import build_policy, build_reference
from csrl.adapters.indic_transcribe import model_settings
from csrl.application.checkpoint import load_checkpoint
from csrl.application.settings import grpo_settings
from csrl.application.training import RunPaths, train_grpo
from csrl.data.manifest import load_manifest
from csrl.infrastructure.config import load_yaml


def main() -> None:
    parser = _cli.run_parser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config_files = [arguments.config, arguments.model_config]
    directory, commit = _cli.start_run(arguments, config_files)
    try:
        settings = grpo_settings(load_yaml(arguments.config))
        model = model_settings(load_yaml(arguments.model_config), require_prompt=True)
        policy = build_policy(model, settings.common)
        load_checkpoint(Path(settings.init_checkpoint), policy.backbone, policy.gate)
        reference = build_reference(policy) if settings.objective.kl_coefficient > 0 else None
        paths = RunPaths(directory, Path(settings.common.manifest_base_directory))
        best = train_grpo(
            policy, reference, settings,
            load_manifest(Path(settings.common.train_manifest)),
            load_manifest(Path(settings.common.dev_manifest)),
            paths,
        )
    except Exception as error:
        _cli.record(arguments, commit, config_files, _cli.STATUS_INVALID, _cli.EMPTY_CELL,
                    _cli.EMPTY_CELL, f"{type(error).__name__}: {error}")
        raise
    _cli.record(arguments, commit, config_files, _cli.STATUS_DONE, f"{best:.4f}",
                _cli.EMPTY_CELL, "best dev MER; SCR in eval_dev.json")


if __name__ == "__main__":
    main()
