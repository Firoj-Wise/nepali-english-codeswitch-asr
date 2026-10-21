"""Shared command line setup for run scripts."""

import argparse
import logging
from pathlib import Path

from csrl.application.run import append_registry_row, git_commit, prepare_run_directory, today
from csrl.domain import constants
from csrl.infrastructure.logging import configure_logging

REPOSITORY = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPOSITORY / "experiments"
REGISTRY = EXPERIMENTS / "registry.md"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_INVALID = "INVALID"
STATUS_TEST = "TEST"
EMPTY_CELL = "-"


def run_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--system", required=True, help="system id from docs/research_plan.md")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="debug runs only; such runs are not reportable")
    return parser


def start_run(arguments: argparse.Namespace, config_files: list[Path]) -> tuple[Path, str]:
    commit = git_commit(REPOSITORY, arguments.allow_dirty)
    directory = prepare_run_directory(EXPERIMENTS, arguments.run_id, config_files)
    configure_logging(logging.INFO, directory / constants.RUN_LOG_FILE_NAME)
    record(arguments, commit, config_files, STATUS_RUNNING, EMPTY_CELL, EMPTY_CELL, EMPTY_CELL)
    return directory, commit


def record(
    arguments: argparse.Namespace,
    commit: str,
    config_files: list[Path],
    status: str,
    mer: str,
    scr: str,
    notes: str,
) -> None:
    dirty = " (dirty)" if arguments.allow_dirty else ""
    append_registry_row(REGISTRY, [
        arguments.run_id, today(), arguments.system, commit[:12] + dirty,
        ", ".join(f.name for f in config_files), status, mer, scr, notes,
    ])
