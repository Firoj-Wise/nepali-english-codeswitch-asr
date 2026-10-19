"""Enumerations shared across the project."""

from enum import Enum


class Script(str, Enum):
    DEVANAGARI = "devanagari"
    LATIN = "latin"
    NUMERIC = "numeric"
    MIXED = "mixed"
    OTHER = "other"


class EditOperation(str, Enum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class DataSource(str, Enum):
    CODESWITCH_INTERNAL = "codeswitch_internal"
    REAL_HUMAN = "real_human"
    SYNTH_INHOUSE_OMNIVOICE = "synth_inhouse_omnivoice"
    SYNTH_BASE_OMNIVOICE = "synth_base_omnivoice"
    SYNTH_GEMINI = "synth_gemini"
    OPENSLR54 = "openslr54"
    FLEURS_NE = "fleurs_ne"
    LIBRISPEECH = "librispeech"


class ManifestField(str, Enum):
    UTTERANCE_ID = "utterance_id"
    AUDIO_PATH = "audio_path"
    DURATION_SECONDS = "duration_seconds"
    SPEAKER_ID = "speaker_id"
    TEXT = "text"
    SOURCE = "source"


class MetricName(str, Enum):
    MER = "mer"
    WER = "wer"
    CER = "cer"
    SCR = "scr"
    SCR_NE = "scr_ne"
    SCR_EN = "scr_en"
    SPER = "sper"
    EMPTY_HYPOTHESIS_COUNT = "empty_hypothesis_count"
    HALLUCINATION_FLAG_COUNT = "hallucination_flag_count"
    UTTERANCE_COUNT = "utterance_count"
    METRIC_VERSION = "metric_version"


class GateClass(int, Enum):
    """Script classes predicted by the script gate. Values are tensor indices."""

    DEVANAGARI = 0
    LATIN = 1
    OTHER = 2


class CheckpointKey(str, Enum):
    BACKBONE = "backbone"
    GATE = "gate"
    STEP = "step"
    DEV_MER = "dev_mer"
    METRIC_VERSION = "metric_version"


class LogKey(str, Enum):
    STEP = "step"
    LOSS = "loss"
    REWARD_TOTAL = "reward_total"
    REWARD_MER = "reward_mer"
    REWARD_SCR = "reward_scr"
    REWARD_HALLUCINATION = "reward_hallucination"
    KL = "kl"
    EMPTY_FRACTION = "empty_fraction"
    ZERO_VARIANCE_GROUPS = "zero_variance_groups"
    CE_LOSS = "ce_loss"
    GATE_LOSS = "gate_loss"


class AdvantageNormalization(str, Enum):
    GROUP_STD = "group_std"  # GRPO: (r - mean) / std
    GROUP_MEAN = "group_mean"  # Dr. GRPO: r - mean, no std division


class LossAggregation(str, Enum):
    SEQUENCE_MEAN = "sequence_mean"  # GRPO: mean over tokens per sequence, then over sequences
    TOKEN_MEAN = "token_mean"  # DAPO: mean over all tokens in the batch
    CONSTANT_NORMALIZER = "constant_normalizer"  # Dr. GRPO: sum over tokens / constant


class ImportanceRatio(str, Enum):
    TOKEN = "token"  # PPO, GRPO, DAPO
    SEQUENCE = "sequence"  # GSPO: length-normalized sequence ratio


class HypothesisField(str, Enum):
    UTTERANCE_ID = "utterance_id"
    REFERENCE = "reference"
    HYPOTHESIS = "hypothesis"


class TtsSystem(str, Enum):
    INHOUSE_OMNIVOICE = "inhouse_omnivoice"
    BASE_OMNIVOICE = "base_omnivoice"
    GEMINI = "gemini"


class DeliveryStyle(str, Enum):
    NEUTRAL = "neutral"
    EXPRESSIVE = "expressive"


class SentenceField(str, Enum):
    SENTENCE_ID = "sentence_id"
    TEXT = "text"
    DOMAIN = "domain"
    SWITCH_TYPE = "switch_type"


class SwitchType(str, Enum):
    SINGLE_WORD = "single_word"
    MULTI_WORD = "multi_word"
    PHRASE = "phrase"


class RejectReason(str, Enum):
    EMPTY = "empty"
    DUPLICATE = "duplicate"
    MIXED_SCRIPT_TOKEN = "mixed_script_token"
    NO_LATIN_TOKEN = "no_latin_token"
    NO_DEVANAGARI_TOKEN = "no_devanagari_token"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    ROUND_TRIP_CER = "round_trip_cer"
    DURATION_SHORT = "duration_short"
    DURATION_LONG = "duration_long"
    TTS_FAILURE = "tts_failure"
    SPEAKING_RATE = "speaking_rate"
    UNKNOWN_ENUM_VALUE = "unknown_enum_value"


class RoundTripScope(str, Enum):
    FULL = "full"
    DEVANAGARI_ONLY = "devanagari_only"
