"""Audio loading and batching. Audio must already be 16 kHz mono; nothing is resampled."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile
import torch

from csrl.domain import constants
from csrl.domain.entities import Utterance
from csrl.domain.errors import AudioFormatError

_FLOAT_DTYPE = "float32"


def load_waveform(path: Path) -> np.ndarray:
    try:
        samples, sample_rate = soundfile.read(str(path), dtype=_FLOAT_DTYPE, always_2d=True)
    except (RuntimeError, soundfile.LibsndfileError) as error:
        raise AudioFormatError(f"cannot read {path}: {error}") from error
    if sample_rate != constants.SAMPLE_RATE_HZ:
        raise AudioFormatError(f"{path}: {sample_rate} Hz, expected {constants.SAMPLE_RATE_HZ}")
    if samples.shape[1] != constants.MONO_CHANNELS:
        raise AudioFormatError(f"{path}: {samples.shape[1]} channels, expected mono")
    return samples[:, 0]


def collate_audio(
    utterances: Sequence[Utterance], base_directory: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-padded waveforms [B, T] and lengths [B]."""
    waveforms = [load_waveform(base_directory / u.audio_path) for u in utterances]
    lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    batch = torch.zeros(len(waveforms), int(lengths.max()))
    for row, waveform in enumerate(waveforms):
        batch[row, : len(waveform)] = torch.from_numpy(waveform)
    return batch.to(device), lengths.to(device)


def write_waveform(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    if sample_rate != constants.SAMPLE_RATE_HZ:
        raise AudioFormatError(f"refusing to write {sample_rate} Hz audio to {path}")
    if waveform.ndim != 1 or waveform.size == 0:
        raise AudioFormatError(f"expected a non-empty mono waveform for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), waveform, sample_rate, subtype=constants.PCM_SUBTYPE)
