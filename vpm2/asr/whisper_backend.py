from pathlib import Path

from vpm2.asr.base import ASRBackend, RawSegment
from vpm2.config import Config


class WhisperBackend(ASRBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._config.asr_model, device="cuda", compute_type="float16",
            )

    def transcribe(self, audio_path: Path) -> list[RawSegment]:
        self._ensure_model()
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self._config.source_lang,
            vad_filter=True,
        )
        return [
            RawSegment(float(s.start), float(s.end), s.text)
            for s in segments
        ]
