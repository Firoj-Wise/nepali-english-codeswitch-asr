"""Experiment bookkeeping per docs/experiment_protocol.md."""

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from csrl.domain import constants
from csrl.domain.errors import CsrlError
from csrl.infrastructure.logging import get_logger

_logger = get_logger(__name__)
_GIT_HEAD = ["git", "rev-parse", "HEAD"]
_GIT_STATUS = ["git", "status", "--porcelain"]
_REGISTRY_COLUMNS = 9


def git_commit(repository: Path, allow_dirty: bool) -> str:
    try:
        commit = subprocess.run(
            _GIT_HEAD, cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            _GIT_STATUS, cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CsrlError(f"cannot read git state in {repository}: {error}") from error
    if status and not allow_dirty:
        raise CsrlError("working tree has uncommitted changes; commit before a reported run")
    if status:
        _logger.warning("running with uncommitted changes; this run is not reportable")
    return commit


def prepare_run_directory(experiments: Path, run_id: str, config_files: list[Path]) -> Path:
    directory = experiments / run_id
    if directory.exists():
        raise CsrlError(f"run directory already exists: {directory}; choose a new run id")
    directory.mkdir(parents=True)
    for config_file in config_files:
        shutil.copyfile(config_file, directory / config_file.name)
    return directory


def append_registry_row(registry: Path, cells: list[str]) -> None:
    if len(cells) != _REGISTRY_COLUMNS:
        raise CsrlError(f"registry rows need {_REGISTRY_COLUMNS} cells, got {len(cells)}")
    if not registry.is_file():
        raise CsrlError(f"registry not found: {registry}")
    row = "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |\n"
    with registry.open("a", encoding=constants.TEXT_ENCODING) as handle:
        handle.write(row)


def today() -> str:
    return datetime.now(UTC).date().isoformat()
