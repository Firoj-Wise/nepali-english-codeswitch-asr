from pathlib import Path

import pytest

from csrl.application.run import append_registry_row, prepare_run_directory
from csrl.data.manifest import load_manifest, verify_frozen, write_manifest
from csrl.domain.entities import Utterance
from csrl.domain.enums import DataSource
from csrl.domain.errors import CsrlError, ManifestSchemaError, SplitIntegrityError


def test_run_directory_is_never_reused(tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text("a: 1\n")
    directory = prepare_run_directory(tmp_path / "exp", "S0-x", [config])
    assert (directory / "c.yaml").is_file()
    with pytest.raises(CsrlError):
        prepare_run_directory(tmp_path / "exp", "S0-x", [config])


def test_registry_append_only_and_column_count(tmp_path: Path) -> None:
    registry = tmp_path / "registry.md"
    registry.write_text("header\n")
    append_registry_row(registry, ["a", "b", "c", "d", "e", "f", "g", "h", "x|y"])
    assert registry.read_text().splitlines() == ["header", "| a | b | c | d | e | f | g | h | x/y |"]
    with pytest.raises(CsrlError):
        append_registry_row(registry, ["too", "few"])


def test_manifest_round_trip_and_frozen_hash(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    utterance = Utterance("u1", "a.wav", 1.5, "s1", "मेरो phone", DataSource.CODESWITCH_INTERNAL)
    write_manifest([utterance], path)
    assert load_manifest(path) == [utterance]
    from csrl.data.manifest import file_sha256

    hash_file = tmp_path / "test.sha256"
    hash_file.write_text(file_sha256(path))
    verify_frozen(path, hash_file)
    path.write_text(path.read_text() + "\n" + path.read_text())
    with pytest.raises(SplitIntegrityError):
        verify_frozen(path, hash_file)


def test_manifest_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"utterance_id": "u1"}\n')
    with pytest.raises(ManifestSchemaError):
        load_manifest(path)
