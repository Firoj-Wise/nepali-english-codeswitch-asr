"""Repository harness. Fails on style violations, banned code patterns, and layer violations."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROSE_SUFFIXES = (".md", ".txt", ".py")
SKIP_PARTS = {".venv", ".git", "__pycache__", "data", "checkpoints", "models", "scratch"}
BANNED_PHRASES_FILE = ROOT / "scripts" / "banned_phrases.txt"
SELF = Path(__file__).resolve()

DASH_PATTERN = re.compile("[–—]")
EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
CODE_PATTERNS = {
    "bare except": re.compile(r"^\s*except\s*:"),
    "swallowed exception": re.compile(r"^\s*except[^\n]*:\s*pass\s*$"),
    "print call": re.compile(r"^\s*print\("),
    "return None": re.compile(r"^\s*return None\s*$"),
}
LAYER_RULES = {
    "domain": ("csrl.data", "csrl.metrics", "csrl.infrastructure", "csrl.application",
               "csrl.adapters", "torch"),
    "metrics": ("csrl.data", "csrl.infrastructure", "csrl.application", "csrl.adapters",
                "torch"),
    "data": ("csrl.application", "csrl.adapters"),
    "infrastructure": ("csrl.application", "csrl.adapters"),
    "application": ("csrl.adapters",),
}


def iter_files() -> list[Path]:
    return [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix in PROSE_SUFFIXES and not SKIP_PARTS & set(p.parts)
    ]


def main() -> int:
    banned = [
        line.strip().lower()
        for line in BANNED_PHRASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures: list[str] = []
    for path in iter_files():
        if path.resolve() in (SELF, BANNED_PHRASES_FILE.resolve()):
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        for number, line in enumerate(text.splitlines(), start=1):
            location = f"{relative}:{number}"
            if DASH_PATTERN.search(line):
                failures.append(f"{location}: em or en dash")
            if EMOJI_PATTERN.search(line):
                failures.append(f"{location}: emoji")
            lowered = line.lower()
            for phrase in banned:
                if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                    failures.append(f"{location}: banned phrase '{phrase}'")
            if path.suffix == ".py":
                for name, pattern in CODE_PATTERNS.items():
                    if pattern.search(line):
                        failures.append(f"{location}: {name}")
        if path.suffix == ".py" and "src" in path.parts:
            layer = path.parent.name
            for forbidden in LAYER_RULES.get(layer, ()):
                if re.search(rf"^\s*(from|import)\s+{re.escape(forbidden)}\b", text, re.MULTILINE):
                    failures.append(f"{relative}: layer {layer} imports {forbidden}")
    for failure in failures:
        sys.stderr.write(failure + "\n")
    sys.stderr.write(f"check_repo: {len(failures)} violation(s)\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
