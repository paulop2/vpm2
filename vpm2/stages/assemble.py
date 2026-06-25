import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from vpm2.artifacts import read_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.timeline import plan_timeline


def _atempo(in_path: Path, out_path: Path, speed: float) -> None:
    # ffmpeg atempo supports 0.5..2.0 per filter; our cap is <=1.25 so one pass.
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-filter:a", f"atempo={speed:.4f}", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


class AssembleStage(Stage):
    name = "assemble"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("06_final.mp4")

    def is_done(self, ctx: Context) -> bool:
        return self.output_path(ctx).exists()

    def run(self, ctx: Context) -> None:
        clips_data = read_json(ctx.path("05_clips.json"))
        sr = int(clips_data["sample_rate"])
        clips_dir = ctx.path("05_clips")
        meta = read_json(ctx.path("01_meta.json"))
        video_duration = float(meta["duration"])

        segs = [{"id": s["id"], "start": s["start"], "end": s["end"],
                 "duration": s["duration"]} for s in clips_data["segments"]]
        placed = plan_timeline(
            segs, video_duration,
            max_speed=ctx.config.max_speed, allow_push=ctx.config.allow_push,
        )
        by_id = {s["id"]: s for s in clips_data["segments"]}

        total_samples = int((video_duration + 5.0) * sr)
        buffer = np.zeros(total_samples, dtype="float32")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for pc in placed:
                src = clips_dir / by_id[pc.id]["clip"]
                if pc.speed > 1.001:
                    sped = tmp / f"{pc.id:04d}_s.wav"
                    _atempo(src, sped, pc.speed)
                    audio, csr = sf.read(str(sped))
                else:
                    audio, csr = sf.read(str(src))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if csr != sr:
                    # resample defensively via ffmpeg-less linear interp
                    idx = np.linspace(0, len(audio) - 1,
                                      int(len(audio) * sr / csr))
                    audio = np.interp(idx, np.arange(len(audio)), audio)
                start_sample = int(pc.start * sr)
                end_sample = start_sample + len(audio)
                if end_sample > len(buffer):
                    buffer = np.concatenate(
                        [buffer, np.zeros(end_sample - len(buffer), "float32")])
                buffer[start_sample:end_sample] += audio.astype("float32")

        pt_wav = ctx.path("06_audio_pt.wav")
        sf.write(str(pt_wav), buffer, sr)

        video = ctx.path("01_video.mp4")
        out = self.output_path(ctx)
        if ctx.config.keep_original_audio:
            cmd = [
                "ffmpeg", "-y", "-i", str(video), "-i", str(pt_wav),
                "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
                "-c:v", "copy", "-c:a", "aac",
                "-disposition:a:0", "default", "-disposition:a:1", "none",
                "-shortest", str(out),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(video), "-i", str(pt_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
            ]
        log = ctx.log_dir() / "assemble.log"
        with open(log, "w") as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
