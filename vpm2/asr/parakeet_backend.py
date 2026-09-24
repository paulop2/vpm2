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


def plan_chunks(total_duration: float, chunk_seconds: int) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < total_duration:
        end = min(start + chunk_seconds, total_duration)
        chunks.append((start, end))
        start = end
    return chunks


def offset_segments(segments: list[RawSegment], offset: float) -> list[RawSegment]:
    return [
        RawSegment(s.start + offset, s.end + offset, s.text)
        for s in segments
    ]


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

        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        duration = len(data) / sr
        chunks = plan_chunks(duration, self._config.parakeet_chunk_seconds)

        self._ensure_model()
        out: list[RawSegment] = []
        for start, end in chunks:
            piece = data[int(start * sr):int(end * sr)]
            if len(piece) == 0:
                continue
            results = self._model.transcribe([piece], timestamps=True)
            if not results:
                continue
            segs = parse_hypothesis(results[0], audio_duration=len(piece) / sr)
            out.extend(offset_segments(segs, start))
        return out
