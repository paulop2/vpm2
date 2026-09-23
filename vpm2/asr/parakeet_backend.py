from pathlib import Path

from vpm2.asr.base import ASRBackend, RawSegment
from vpm2.config import Config


def parse_hypothesis(hyp, audio_duration: float | None = None) -> list[RawSegment]:
    timestamp = getattr(hyp, "timestamp", None) or {}
    items = timestamp.get("segment") or []
    if items:
        return [
            RawSegment(float(t["start"]), float(t["end"]), str(t["segment"]))
            for t in items
        ]
    text = (getattr(hyp, "text", "") or "").strip()
    if not text:
        return []
    return [RawSegment(0.0, float(audio_duration or 0.0), text)]


class ParakeetBackend(ASRBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import nemo.collections.asr as nemo_asr
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._config.parakeet_model,
            )
            self._model.eval()

    def transcribe(self, audio_path: Path) -> list[RawSegment]:
        import soundfile as sf

        self._ensure_model()
        results = self._model.transcribe([str(audio_path)], timestamps=True)
        if not results:
            return []
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate
        return parse_hypothesis(results[0], audio_duration=duration)
