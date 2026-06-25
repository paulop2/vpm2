from pathlib import Path

from faster_whisper import WhisperModel

from vpm2.artifacts import valid_transcript, write_json
from vpm2.context import Context
from vpm2.stages.base import Stage


class TranscribeStage(Stage):
    name = "transcribe"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("03_transcript.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_transcript(self.output_path(ctx))

    def run(self, ctx: Context) -> None:
        model = WhisperModel(
            ctx.config.asr_model, device="cuda", compute_type="float16",
        )
        segments, _info = model.transcribe(
            str(ctx.path("02_audio.wav")),
            language=ctx.config.source_lang,
            vad_filter=True,
        )
        out_segments = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if not text:
                continue
            out_segments.append({
                "id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
        write_json(self.output_path(ctx), {
            "language": ctx.config.source_lang,
            "segments": out_segments,
        })
        del model
