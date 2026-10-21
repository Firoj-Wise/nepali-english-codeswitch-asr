"""Small shared helpers for data scripts."""

import json
from pathlib import Path

from csrl.domain import constants
from csrl.domain.entities import Rejection


def write_rejections(path: Path, rejections: list[Rejection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.__dict__, ensure_ascii=False) for r in rejections]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding=constants.TEXT_ENCODING)


def with_suffix_tag(path: Path, tag: str) -> Path:
    """data/x.jsonl with tag 'filtered' becomes data/x.filtered.jsonl."""
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")
