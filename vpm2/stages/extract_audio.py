import subprocess
from pathlib import Path

from vpm2.context import Context
from vpm2.stages.base import Stage


class ExtractAudioStage(Stage):
    name = "extract_audio"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("02_audio.wav")

    def is_done(self, ctx: Context) -> bool:
        return self.output_path(ctx).exists()

    def run(self, ctx: Context) -> None:
        video = ctx.path("01_video.mp4")
        out = self.output_path(ctx)
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out),
        ]
        log = ctx.log_dir() / "extract_audio.log"
        with ctx.reporter.spinner("extraindo áudio (16kHz mono)"):
            with open(log, "w") as lf:
                subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)