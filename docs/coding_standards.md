# Coding Standards

## 1. Architecture

Clean architecture with dependencies pointing inward.

| Layer | Path | May import | Must not import |
| --- | --- | --- | --- |
| Domain | `csrl/domain` | standard library | anything else in `csrl` |
| Metrics | `csrl/metrics` | domain | data, infrastructure, application |
| Data | `csrl/data` | domain | application |
| Infrastructure | `csrl/infrastructure` | domain | application |
| Application | `csrl/application` | domain, metrics, data, infrastructure, torch | adapters |
| Adapters | `csrl/adapters` | everything | nothing imports adapters except scripts |

NeMo code lives only in `csrl/adapters`. Application code depends on `BackbonePort` in `application/ports.py`, never on NeMo. Torch is not imported by `domain` or `metrics`.

`scripts/check_repo.py` enforces the import rules.

## 2. No magic values

- Literals with meaning live in `domain/constants.py` or as enums in `domain/enums.py`.
- Tunable values (learning rates, lambdas, group size, paths) live in `configs/*.yaml` and are loaded into typed dataclasses.
- Allowed inline literals: 0, 1, -1 as index or identity, empty string as a join separator, and format strings for logging.
- String keys for dictionaries and JSON fields come from enums or constants, never typed inline.

## 3. Errors and fallbacks

- Every failure raises a subclass of `CsrlError` from `domain/errors.py`.
- Forbidden: bare `except`, `except Exception` without re-raise, `pass` in an except block, returning `None` or an empty container to signal failure, `dict.get(key, default)` for required data, silent type coercion.
- Errors carry context: what was being processed and which value was wrong.

## 4. Returns

- Every function has type hints and a non-optional return type unless absence is a documented, legitimate outcome, in which case use an explicit result type.
- No function ends without returning or raising.

## 5. Logging

- Use `get_logger(__name__)` from `csrl.infrastructure.logging`. No `print`.
- Log at INFO for stage boundaries, DEBUG for per-item detail, WARNING for excluded data with counts, ERROR before raising at application boundaries.
- Log messages state facts with values: `excluded 12 clips longer than 30.0 s`.

## 6. Determinism

- Seeds come from config and are set through one function. Every run logs its seed.
- Metric functions are pure and deterministic.

## 7. Tests

- `pytest`. Metric tests use hand-computed expected values with the arithmetic written in a comment.
- Tests for error paths: each typed error has at least one test that triggers it.

## 8. Style

- `ruff` for lint and format, `mypy --strict` for `domain` and `metrics`.
- Names are descriptive. No single-letter names outside short comprehensions and math with documented symbols.
