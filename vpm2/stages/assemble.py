import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from vpm2.artifacts import read_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.timeline import plan_timeline


def _tmp_sibling(dest: Path) -> Path:
    # Sibling temp in the same directory keeps os.replace atomic; keeping the
    # original extension lets ffmpeg/soundfile infer the container format.
    return dest.with_name(f".{dest.stem}.tmp{dest.suffix}")


def _write_wav_atomic(dest: Path, audio, sr: int) -> None:
    tmp = _tmp_sibling(dest)
    sf.write(str(tmp), audio, sr, format="WAV")
    os.replace(tmp, dest)


def _atempo(in_path: Path, out_path: Path, speed: float) -> None:
    # ffmpeg atempo supports 0.5..2.0 per filter; our cap is <=2.0 so one pass.
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

        with tempfile.TemporaryDirectory() as tmp, \
                ctx.reporter.bar("montando trilha PT-BR", total=len(placed)) as bar:
            tmp = Path(tmp)
            # Time-stretching is an independent ffmpeg subprocess per clip (releases
            # the GIL), so stretch every overrun clip up front in parallel; the
            # buffer summation below stays sequential to keep overlap-add correct.
            sped: dict[int, Path] = {}
            to_stretch = [pc for pc in placed if pc.speed > 1.001]
            if to_stretch:
                workers = min(len(to_stretch), (os.cpu_count() or 4))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = []
                    for pc in to_stretch:
                        dst = tmp / f"{pc.id:04d}_s.wav"
                        sped[pc.id] = dst
                        futs.append(ex.submit(
                            _atempo, clips_dir / by_id[pc.id]["clip"], dst, pc.speed))
                    for f in futs:
                        f.result()  # surface ffmpeg failures

            for pc in placed:
                bar.advance()
                read_path = sped.get(pc.id) or clips_dir / by_id[pc.id]["clip"]
                audio, csr = sf.read(str(read_path))
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
        _write_wav_atomic(pt_wav, buffer, sr)

        video = ctx.path("01_video.mp4")
        out = self.output_path(ctx)
        # Mux to a hidden sibling first: an interrupted ffmpeg would otherwise
        # leave a partial 06_final.mp4 that a later resume treats as finished.
        tmp_out = _tmp_sibling(out)
        if ctx.config.keep_original_audio:
            cmd = [
                "ffmpeg", "-y", "-i", str(video), "-i", str(pt_wav),
                "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
                "-c:v", "copy", "-c:a", "aac",
                "-disposition:a:0", "default", "-disposition:a:1", "none",
                "-shortest", str(tmp_out),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(video), "-i", str(pt_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp_out),
            ]
        log = ctx.log_dir() / "assemble.log"
        with ctx.reporter.spinner("muxando vídeo + áudio PT-BR"), open(log, "w") as lf:
            try:
                subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
            except BaseException:
                tmp_out.unlink(missing_ok=True)
                raise
            os.replace(tmp_out, out)
