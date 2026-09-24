from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from vpm2.config import Config


@dataclass(frozen=True)
class RawSegment:
    start: float
    end: float
    text: str


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[RawSegment]:
        ...


def get_asr_backend(config: Config) -> ASRBackend:
    if config.asr_backend == "parakeet":
        from vpm2.asr.parakeet_backend import ParakeetBackend
        return ParakeetBackend(config)
    if config.asr_backend == "whisper":
        from vpm2.asr.whisper_backend import WhisperBackend
        return WhisperBackend(config)
    raise ValueError(f"unknown asr backend: {config.asr_backend}")
