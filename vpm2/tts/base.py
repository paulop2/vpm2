from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from vpm2.config import Config


class TTSBackend(ABC):
    sample_rate: int = 24000

    @abstractmethod
    def synth(self, text: str, ref_wav: Path | None) -> np.ndarray:
        """Return float32 mono audio at self.sample_rate."""


def get_backend(config: Config) -> TTSBackend:
    if config.tts_backend == "chatterbox":
        from vpm2.tts.chatterbox_backend import ChatterboxBackend
        return ChatterboxBackend(config)
    raise ValueError(f"unknown tts backend: {config.tts_backend}")
