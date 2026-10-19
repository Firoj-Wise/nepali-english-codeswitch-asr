from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from csrl.application.quality_filter import FilterSettings, filter_synthetic, scoped_text
from csrl.application.synthesis import (
    SynthesisSettings,
    plan_assignments,
    run_synthesis,
    sentence_id_of,
)
from csrl.data.augment import TelephonySettings, mu_law_round_trip, resample, telephony
from csrl.data.leakage import check_leakage
from csrl.data.manifest import load_manifest
from csrl.data.sentences import SentenceLimits, validate_sentences
from csrl.data.splits import split_by_speaker
from csrl.domain.entities import SentenceRecord, Utterance, VoiceSpec
from csrl.domain.enums import DataSource, RejectReason, RoundTripScope, Split
from csrl.domain.errors import ConfigError, ManifestSchemaError, SynthesisBudgetError, TtsError
from csrl.infrastructure.audio import load_waveform

LIMITS = SentenceLimits(min_words=2, max_words=10)


def _sentence(index: int, text: str) -> SentenceRecord:
    return SentenceRecord(f"s{index}", text, "banking", "single_word")


def _voice(name: str) -> VoiceSpec:
    return VoiceSpec("gemini", name, "neutral", None, None)


def test_sentence_validation_reasons() -> None:
    records = [
        _sentence(0, "मेरो phone राम्रो छ"),
        _sentence(1, "मेरो फोन राम्रो छ"),  # no Latin word
        _sentence(2, "my phone is good"),  # no Devanagari word
        _sentence(3, "phoneमा राम्रो"),  # attached suffix is split at the script boundary
        _sentence(4, "मेरो Phone राम्रो छ।"),  # duplicate after normalization
        _sentence(5, "phone"),  # too short and no Devanagari
        SentenceRecord("s6", "मेरो phone", "banking", "not_a_type"),
    ]
    result = validate_sentences(records, LIMITS)
    assert [r.sentence_id for r in result.valid] == ["s0", "s3"]
    reasons = {r.item_id: r.reason for r in result.rejected}
    assert reasons == {
        "s1": RejectReason.NO_LATIN_TOKEN.value,
        "s2": RejectReason.NO_DEVANAGARI_TOKEN.value,
        "s4": RejectReason.DUPLICATE.value,
        "s5": RejectReason.NO_DEVANAGARI_TOKEN.value,
        "s6": RejectReason.UNKNOWN_ENUM_VALUE.value,
    }


def test_duplicate_sentence_id_raises() -> None:
    with pytest.raises(ManifestSchemaError):
        validate_sentences([_sentence(0, "मेरो phone"), _sentence(0, "तिम्रो phone")], LIMITS)


def test_leakage_exact_and_ngram() -> None:
    report = check_leakage(
        ["मेरो phone राम्रो छ", "आज bank जानु छ भोलि होइन", "unrelated नेपाली text यहाँ"],
        ["मेरो Phone राम्रो छ।", "म आज bank जानु छ"],
        ngram_size=3,
    )
    assert report.exact_matches == ["मेरो phone राम्रो छ"]
    assert len(report.ngram_overlaps) == 1
    text, shared = report.ngram_overlaps[0]
    assert text == "आज bank जानु छ भोलि होइन"
    assert shared in {"आज bank जानु", "bank जानु छ"}
    assert not report.is_clean
    assert check_leakage(["a b c"], ["x y z"], 3).is_clean


def test_plan_is_deterministic_and_balanced() -> None:
    sentences = [_sentence(i, "मेरो phone") for i in range(40)]
    voices = [_voice(v) for v in ("A", "B", "C", "D")]
    settings = SynthesisSettings(voices_per_sentence=2, max_failure_fraction=0.1, seed=3)
    first = plan_assignments(sentences, voices, settings)
    assert first == plan_assignments(sentences, voices, settings)
    assert len(first) == 80
    for index in range(40):
        pair = {a.voice.voice_id for a in first[2 * index : 2 * index + 2]}
        assert len(pair) == 2
    counts = {v: sum(a.voice.voice_id == v for a in first) for v in "ABCD"}
    assert all(count > 0 for count in counts.values())
    with pytest.raises(ConfigError):
        plan_assignments(sentences, voices[:1], settings)


class FakeTts:
    def __init__(self, fail_texts: set[str]) -> None:
        self.calls = 0
        self.fail_texts = fail_texts

    @property
    def data_source(self) -> DataSource:
        return DataSource.SYNTH_GEMINI

    def synthesize(self, text: str, voice: VoiceSpec) -> tuple[np.ndarray, int]:
        self.calls += 1
        if text in self.fail_texts:
            raise TtsError("simulated failure")
        return np.full(24000, 0.1, dtype=np.float32), 24000


def test_synthesis_writes_resumes_and_records_failures(tmp_path: Path) -> None:
    base = tmp_path / "data"
    sentences = [_sentence(i, f"मेरो phone {i}") for i in range(5)]
    settings = SynthesisSettings(voices_per_sentence=1, max_failure_fraction=0.5, seed=0)
    plan = plan_assignments(sentences, [_voice("A"), _voice("B")], settings)
    manifest = base / "manifests" / "m.jsonl"
    tts = FakeTts(fail_texts={"मेरो phone 3"})
    written = run_synthesis(tts, plan, base / "synthetic" / "g", base, manifest, settings)
    assert written == 4
    utterances = load_manifest(manifest)
    assert len(utterances) == 4
    first = utterances[0]
    assert first.audio_path.startswith("synthetic/g/")
    audio = load_waveform(base / first.audio_path)
    assert len(audio) == 16000  # 1 s at 24 kHz resampled to 16 kHz
    assert first.duration_seconds == pytest.approx(1.0)
    assert (base / "synthetic" / "g" / "failures.jsonl").read_text().count("\n") == 1
    again = FakeTts(fail_texts=set())
    assert run_synthesis(again, plan, base / "synthetic" / "g", base, manifest, settings) == 1
    assert again.calls == 1


def test_synthesis_failure_budget(tmp_path: Path) -> None:
    base = tmp_path / "data"
    sentences = [_sentence(i, "मेरो phone") for i in range(4)]
    settings = SynthesisSettings(voices_per_sentence=1, max_failure_fraction=0.2, seed=0)
    plan = plan_assignments(sentences, [_voice("A")], settings)
    with pytest.raises(SynthesisBudgetError):
        run_synthesis(FakeTts({"मेरो phone"}), plan, base / "s", base, base / "m.jsonl", settings)


def test_synthesis_output_must_be_under_base(tmp_path: Path) -> None:
    settings = SynthesisSettings(voices_per_sentence=1, max_failure_fraction=0.2, seed=0)
    plan = plan_assignments([_sentence(0, "मेरो phone")], [_voice("A")], settings)
    with pytest.raises(ConfigError):
        run_synthesis(FakeTts(set()), plan, tmp_path / "elsewhere", tmp_path / "data",
                      tmp_path / "m.jsonl", settings)


def test_synthetic_split_keeps_sentences_together() -> None:
    utterances = [
        Utterance(f"s{i}__gemini-{v}-neutral", "x.wav", 1.0, f"gemini:{v}:neutral",
                  "मेरो phone", DataSource.SYNTH_GEMINI)
        for i in range(30) for v in ("A", "B")
    ]
    result = split_by_speaker(utterances, {Split.TRAIN: 0.8, Split.DEV: 0.2}, seed=1,
                              group_key=sentence_id_of)
    train_ids = {sentence_id_of(u) for u in result.utterances[Split.TRAIN]}
    dev_ids = {sentence_id_of(u) for u in result.utterances[Split.DEV]}
    assert train_ids and dev_ids and not train_ids & dev_ids
    assert Split.TEST not in result.utterances


class FakeTranscriber:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs

    def transcribe(self, utterances: Sequence[Utterance]) -> list[str]:
        return [self.outputs[u.utterance_id] for u in utterances]


def _utt(uid: str, text: str, seconds: float) -> Utterance:
    return Utterance(uid, f"{uid}.wav", seconds, "g:A:neutral", text, DataSource.SYNTH_GEMINI)


def test_scoped_text_devanagari_only() -> None:
    assert scoped_text("मेरो phone राम्रो", RoundTripScope.DEVANAGARI_ONLY) == "मेरो राम्रो"
    assert scoped_text("मेरो phone", RoundTripScope.FULL) == "मेरो phone"


def test_filter_reasons() -> None:
    settings = FilterSettings(
        max_round_trip_cer=0.2, min_duration_seconds=0.5, max_duration_seconds=10.0,
        scope=RoundTripScope.DEVANAGARI_ONLY, min_characters_per_second=2.0,
        max_characters_per_second=30.0,
    )
    text = "मेरो phone राम्रो छ"  # 15 characters without spaces
    utterances = [
        _utt("good", text, 2.0),  # ASR writes phone in Devanagari: ignored by scope
        _utt("bad", text, 2.0),
        _utt("short", text, 0.1),
        _utt("slow", text, 9.9),  # 15 / 9.9 = 1.5 chars per second
    ]
    transcriber = FakeTranscriber({"good": "मेरो फोन राम्रो छ", "bad": "तिम्रो किताब"})
    result = filter_synthetic(utterances, transcriber, settings)
    assert [u.utterance_id for u in result.kept] == ["good"]
    reasons = {r.item_id: r.reason for r in result.rejected}
    assert reasons == {
        "bad": RejectReason.ROUND_TRIP_CER.value,
        "short": RejectReason.DURATION_SHORT.value,
        "slow": RejectReason.SPEAKING_RATE.value,
    }


def test_resample_and_mu_law() -> None:
    tone = np.sin(np.linspace(0, 200 * np.pi, 24000)).astype(np.float32) * 0.5
    assert len(resample(tone, 24000, 16000)) == 16000
    round_trip = mu_law_round_trip(tone, 256)
    assert float(np.max(np.abs(round_trip - tone))) < 0.02


def test_telephony_is_deterministic_and_bounded() -> None:
    waveform = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
    settings = TelephonySettings(snr_db_low=10.0, snr_db_high=20.0, seed=5)
    first = telephony(waveform, "u1", settings, 256)
    assert np.array_equal(first, telephony(waveform, "u1", settings, 256))
    assert not np.array_equal(first, telephony(waveform, "u2", settings, 256))
    assert first.shape == waveform.shape and float(np.max(np.abs(first))) <= 1.0
