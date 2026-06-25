from pathlib import Path

import soundfile as sf

from vpm2.artifacts import read_json, valid_clips, write_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.tts.base import get_backend
from vpm2.voice_sample import pick_reference_window


def _extract_reference(ctx: Context) -> Path:
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    audio_path = ctx.path("02_audio.wav")
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    # faster-whisper VAD expects 16kHz float32; timestamps come back in SAMPLES.
    ts = get_speech_timestamps(
        data.astype("float32"),
        vad_options=VadOptions(),
        sampling_rate=sr,
    )
    spans = [(t["start"] / sr, t["end"] / sr) for t in ts]
    win = pick_reference_window(spans)
    ref_path = ctx.path("ref_voice.wav")
    if win is None:
        # fallback: first 10s
        start, end = 0.0, min(10.0, len(data) / sr)
    else:
        start, end = win
    sf.write(str(ref_path), data[int(start * sr):int(end * sr)], sr)
    return ref_path


class SynthesizeStage(Stage):
    name = "synthesize"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("05_clips.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_clips(self.output_path(ctx), ctx.path("05_clips"))

    def run(self, ctx: Context) -> None:
        clips_dir = ctx.path("05_clips")
        clips_dir.mkdir(parents=True, exist_ok=True)

        if ctx.config.voice_mode == "cloning":
            with ctx.reporter.spinner("extraindo voz de referência do vídeo"):
                ref = _extract_reference(ctx)
        elif ctx.config.voice_mode == "preset":
            if not ctx.config.preset_ref_wav:
                raise ValueError(
                    "voice_mode='preset' requires a reference clip. "
                    "Pass --preset-ref <clean_pt_voice.wav> "
                    "(Chatterbox has no built-in preset voices)."
                )
            ref = Path(ctx.config.preset_ref_wav)
            if not ref.exists():
                raise FileNotFoundError(f"preset_ref_wav not found: {ref}")
        else:
            raise ValueError(f"unknown voice_mode: {ctx.config.voice_mode}")

        with ctx.reporter.spinner("carregando modelo de voz (Chatterbox TTS)"):
            backend = get_backend(ctx.config)
        segs = read_json(ctx.path("04_translation.json"))["segments"]
        out = []
        with ctx.reporter.bar("sintetizando voz PT-BR", total=len(segs)) as bar:
            for s in segs:
                audio = backend.synth(s["text_pt"], ref)
                name = f"{s['id']:04d}.wav"
                sf.write(str(clips_dir / name), audio, backend.sample_rate)
                out.append({
                    "id": s["id"], "start": s["start"], "end": s["end"],
                    "clip": name, "duration": len(audio) / backend.sample_rate,
                })
                bar.advance()
        write_json(self.output_path(ctx), {
            "sample_rate": backend.sample_rate, "segments": out,
        })
