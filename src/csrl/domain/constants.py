"""Named constants. Changing a metric constant requires a METRIC_VERSION bump."""

from typing import Final

from csrl.domain.enums import Script

METRIC_VERSION: Final[int] = 1

DEVANAGARI_BLOCK_START: Final[int] = 0x0900
DEVANAGARI_BLOCK_END: Final[int] = 0x097F

DEVANAGARI_DIGITS: Final[str] = "०१२३४५६७८९"
ASCII_DIGITS: Final[str] = "0123456789"

DEVANAGARI_DANDA: Final[str] = "।"
DEVANAGARI_DOUBLE_DANDA: Final[str] = "॥"
PUNCTUATION_TO_STRIP: Final[str] = (
    "!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~" + DEVANAGARI_DANDA + DEVANAGARI_DOUBLE_DANDA
)

UNICODE_NORMAL_FORM: Final = "NFC"
TOKEN_SEPARATOR: Final[str] = " "

CONFUSABLE_SCRIPTS: Final[frozenset[Script]] = frozenset({Script.DEVANAGARI, Script.LATIN})

HALLUCINATION_LENGTH_RATIO: Final[float] = 1.5
REPEAT_NGRAM_SIZE: Final[int] = 3
REPEAT_MIN_COUNT: Final[int] = 3

BOOTSTRAP_SAMPLES: Final[int] = 1000
CONFIDENCE_LEVEL: Final[float] = 0.95

REWARD_MER_CLIP: Final[float] = 1.0

SCR_HEADROOM_THRESHOLD: Final[float] = 0.01

MAX_CLIP_SECONDS: Final[float] = 30.0

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"

SAMPLE_RATE_HZ: Final[int] = 16000
MONO_CHANNELS: Final[int] = 1

SENTENCEPIECE_WORD_MARKER: Final[str] = "▁"

ADVANTAGE_EPSILON: Final[float] = 1.0e-6

BEST_CHECKPOINT_NAME: Final[str] = "best.pt"
LAST_CHECKPOINT_NAME: Final[str] = "last.pt"
EVAL_FILE_TEMPLATE: Final[str] = "eval_{split}.json"
HYPOTHESES_FILE_TEMPLATE: Final[str] = "hyps_{split}.jsonl"
RUN_LOG_FILE_NAME: Final[str] = "run.log"
CONFIG_COPY_FILE_NAME: Final[str] = "config.yaml"
TEST_HASH_FILE_NAME: Final[str] = "test.sha256"
SPEAKER_LIST_SUFFIX: Final[str] = ".speakers.txt"
MANIFEST_SUFFIX: Final[str] = ".jsonl"
TEXT_ENCODING: Final[str] = "utf-8"

UNVERIFIED_BINDING_NOTICE: Final[str] = (
    "indic-transcribe binding is unverified; run scripts/smoke_test_model.py first"
)

SECONDS_PER_HOUR: Final[int] = 3600

SYNTHESIS_ID_SEPARATOR: Final[str] = "__"
FAILURES_FILE_NAME: Final[str] = "failures.jsonl"
REJECTED_FILE_NAME: Final[str] = "rejected.jsonl"
WAV_SUFFIX: Final[str] = ".wav"
PCM_SUBTYPE: Final[str] = "PCM_16"
TELEPHONY_SAMPLE_RATE_HZ: Final[int] = 8000
MU_LAW_MU: Final[int] = 255
DECIBEL_POWER_FACTOR: Final[float] = 10.0
SPEAKER_ID_SEPARATOR: Final[str] = ":"
FILENAME_SAFE_SEPARATOR: Final[str] = "-"

SCRIPT_BOUNDARY_PATTERN: Final[str] = (
    "(?<=[a-z0-9])(?=[\u0900-\u097F])|(?<=[\u0900-\u097F])(?=[a-z0-9])"
)
