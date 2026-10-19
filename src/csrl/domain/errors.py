"""Typed errors. Code raises these instead of returning fallback values."""


class CsrlError(Exception):
    """Base class for all project errors."""


class EmptyReferenceError(CsrlError):
    """A reference transcript is empty after normalization."""


class ManifestSchemaError(CsrlError):
    """A manifest line is missing a field or has a wrong type."""


class ConfigError(CsrlError):
    """A configuration file is missing a key or has an invalid value."""


class LengthMismatchError(CsrlError):
    """Paired sequences have different lengths."""


class SplitIntegrityError(CsrlError):
    """The test manifest hash does not match the frozen hash."""


class SplitError(CsrlError):
    """Speaker-disjoint splitting failed its constraints."""


class AudioFormatError(CsrlError):
    """Audio is not 16 kHz mono or cannot be read."""


class RunHaltedError(CsrlError):
    """A training run hit a stopping rule from docs/research_plan.md."""


class BindingNotImplementedError(CsrlError):
    """The model binding has not been completed for this backbone."""


class ModelEquivalenceError(CsrlError):
    """The repository decoding path disagrees with the official model output."""


class TtsError(CsrlError):
    """A TTS backend failed to produce audio for one request."""


class LeakageError(CsrlError):
    """Training text overlaps with evaluation text."""


class SynthesisBudgetError(CsrlError):
    """Synthesis failures exceeded the configured tolerance."""
