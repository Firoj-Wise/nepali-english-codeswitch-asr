"""Interfaces that application code depends on. Framework adapters implement them."""

from collections.abc import Sequence
from typing import Protocol

from csrl.domain.entities import Utterance


class Transcriber(Protocol):
    def transcribe(self, utterances: Sequence[Utterance]) -> list[str]:
        """Return one hypothesis per utterance, in input order."""
        ...
