import os
from pathlib import Path

import soundfile as sf

from vpm2.artifacts import read_json, valid_clips, write_json
from vpm2.context import Context
from vpm2.gpu import free_cuda
from vpm2.stages.base import Stage
from vpm2.tts.base import get_backend
from vpm2.voice_sample import pick_reference_window


def _write_wav_atomic(dest: Path, audio, sr: int) -> None:
    # Write to a sibling temp file then rename: os.replace is atomic on the same
    # filesystem, so an interrupted run can never leave a half-written clip that
    # a later resume would mistake for finished work.
    tmp = dest.with_name(f".{dest.name}.tmp")
    # explicit format: the temp name's .tmp suffix hides the wav extension that
    # soundfile would otherwise infer.
    sf.write(str(tmp), audio, sr, format="WAV")
    os.replace(tmp, dest)


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

    def _resolve_reference(self, ctx: Context) -> Path:
        if ctx.config.voice_mode == "cloning":
            with ctx.reporter.spinner("extraindo voz de referência do vídeo"):
                return _extract_reference(ctx)
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
            return ref
        raise ValueError(f"unknown voice_mode: {ctx.config.voice_mode}")

    def run(self, ctx: Context) -> None:
        clips_dir = ctx.path("05_clips")
        clips_dir.mkdir(parents=True, exist_ok=True)
        segs = read_json(ctx.path("04_translation.json"))["segments"]

        def clip_path(seg) -> Path:
            return clips_dir / f"{seg['id']:04d}.wav"

        # Resume: a clip already on disk is complete (clips are written
        # atomically), so reuse it. Only spin up the reference + TTS model when
        # something is actually missing -- a full resume after a crash that only
        # lost the manifest costs no model load and no GPU.
        pending = [s for s in segs if not clip_path(s).exists()]
        backend = ref = None
        sample_rate = None
        if pending:
            ref = self._resolve_reference(ctx)
            with ctx.reporter.spinner("carregando modelo de voz (Chatterbox TTS)"):
                backend = get_backend(ctx.config)
            sample_rate = backend.sample_rate

        out = []
        with ctx.reporter.bar("sintetizando voz PT-BR", total=len(segs)) as bar:
            for s in segs:
                dest = clip_path(s)
                if dest.exists():
                    sf_info = sf.info(str(dest))
                    if sample_rate is None:
                        sample_rate = sf_info.samplerate
                    duration = sf_info.frames / sf_info.samplerate
                else:
                    audio = backend.synth(s["text_pt"], ref)
                    _write_wav_atomic(dest, audio, backend.sample_rate)
                    duration = len(audio) / backend.sample_rate
                out.append({
                    "id": s["id"], "start": s["start"], "end": s["end"],
                    "clip": dest.name, "duration": duration,
                })
                bar.advance()
        if backend is not None:
            del backend
            free_cuda()
        write_json(self.output_path(ctx), {
            "sample_rate": sample_rate, "segments": out,
        })
