from pathlib import Path

from vpm2.artifacts import valid_transcript, write_json
from vpm2.asr.base import get_asr_backend
from vpm2.asr.normalize import normalize_segments
from vpm2.context import Context
from vpm2.gpu import free_cuda
from vpm2.stages.base import Stage


class TranscribeStage(Stage):
    name = "transcribe"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("03_transcript.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_transcript(self.output_path(ctx))

    def run(self, ctx: Context) -> None:
        backend = get_asr_backend(ctx.config)
        with ctx.reporter.spinner(f"transcrevendo ({ctx.config.asr_backend})"):
            raw = backend.transcribe(ctx.path("02_audio.wav"))
        segments = normalize_segments(raw)
        write_json(self.output_path(ctx), {
            "language": ctx.config.source_lang,
            "segments": segments,
        })
        del backend
        free_cuda()
