"""Write the facts needed to finish the binding to experiments/model_inspection.txt.

Records: module tree of the NeMo model, tokenizer special tokens, and every source line in
the shipped model code that mentions prompts, modes, or task tokens. An agent reads this
file to fill prompt_token_ids in configs/model.yaml. Nothing here changes any config.
"""

import argparse
import logging
import re
from pathlib import Path

from csrl.adapters.indic_transcribe import NemoCanaryBackbone, model_settings
from csrl.domain import constants
from csrl.infrastructure.config import load_yaml
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)
_OUTPUT = Path(__file__).resolve().parent.parent / "experiments" / "model_inspection.txt"
_SOURCE_PATTERN = re.compile(r"prompt|mode|mixed|task|source_lang|target_lang|pnc|itn", re.IGNORECASE)
_SPECIAL_PATTERN = re.compile(r"^<\|.*\|>$|^<.*>$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    arguments = parser.parse_args()
    configure_logging(logging.INFO, None)
    settings = model_settings(load_yaml(arguments.model_config), require_prompt=False)
    lines: list[str] = ["# model inspection", ""]
    for source in sorted(settings.model_directory.rglob("*.py")):
        lines.append(f"## {source.relative_to(settings.model_directory)}")
        for number, text in enumerate(source.read_text(encoding=constants.TEXT_ENCODING)
                                      .splitlines(), start=1):
            if _SOURCE_PATTERN.search(text):
                lines.append(f"{number}: {text}")
        lines.append("")
    backbone = NemoCanaryBackbone.load(settings)
    lines.append("## special tokens (id, piece)")
    for index, piece in enumerate(backbone.tokenizer.vocabulary):
        if _SPECIAL_PATTERN.match(piece):
            lines.append(f"{index}: {piece}")
    lines.append("")
    lines.append("## module tree")
    lines.append(repr(backbone._model))
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text("\n".join(lines) + "\n", encoding=constants.TEXT_ENCODING)
    _logger.info("wrote %s", _OUTPUT)


if __name__ == "__main__":
    main()
