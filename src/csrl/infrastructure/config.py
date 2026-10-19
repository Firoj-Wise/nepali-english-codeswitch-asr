"""YAML config loading. Missing keys raise ConfigError."""

from pathlib import Path
from typing import Any

import yaml

from csrl.domain.errors import ConfigError

_ENCODING = "utf-8"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    content = yaml.safe_load(path.read_text(encoding=_ENCODING))
    if not isinstance(content, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return content


def require(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ConfigError(f"missing config key: {key}")
    return config[key]
