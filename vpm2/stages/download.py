from pathlib import Path

import yt_dlp

from vpm2.artifacts import write_json
from vpm2.context import Context
from vpm2.stages.base import Stage


class DownloadStage(Stage):
    name = "download"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("01_video.mp4")

    def is_done(self, ctx: Context) -> bool:
        return self.output_path(ctx).exists() and ctx.path("01_meta.json").exists()

    def run(self, ctx: Context) -> None:
        out = self.output_path(ctx)
        opts = {
            "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(out),
            "quiet": True,
            "noprogress": True,
        }
        with ctx.reporter.spinner("baixando vídeo do YouTube"):
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(ctx.url, download=True)
        write_json(ctx.path("01_meta.json"), {
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "url": ctx.url,
            "duration": float(info.get("duration") or 0.0),
        })