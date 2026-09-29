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
        backend = ctx.config.asr_backend
        try:
            asr = get_asr_backend(ctx.config)
            with ctx.reporter.spinner(f"transcrevendo ({backend})"):
                raw = asr.transcribe(ctx.path("02_audio.wav"))
        except ImportError as exc:
            raise SystemExit(
                f"[vpm2] dependências do backend '{backend}' não estão instaladas "
                f"({exc}). Instale-as ou rode com --asr-backend whisper."
            ) from exc
        except RuntimeError as exc:
            detail = str(exc)
            if "cuda" in detail.lower() or "out of memory" in detail.lower():
                raise SystemExit(
                    f"[vpm2] o backend '{backend}' falhou com um erro de GPU/CUDA "
                    f"({detail}). Rode com --asr-backend whisper para usar a CPU."
                ) from exc
            raise
        segments = normalize_segments(raw)
        write_json(self.output_path(ctx), {
            "language": ctx.config.source_lang,
            "segments": segments,
        })
        del asr
        free_cuda()
