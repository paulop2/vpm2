import numpy as np
import soundfile as sf

from vpm2.artifacts import read_json, valid_clips, write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import synthesize as synth_mod
from vpm2.stages.synthesize import SynthesizeStage

SR = 24000


class FakeBackend:
    sample_rate = SR

    def __init__(self):
        self.calls = []

    def synth(self, text, ref):
        self.calls.append(text)
        # half a second of silence, deterministic length
        return np.zeros(SR // 2, dtype="float32")


def make_ctx(tmp_path):
    # preset mode with a real ref wav avoids the VAD-based reference extraction
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(SR, dtype="float32"), SR)
    cfg = Config(voice_mode="preset", preset_ref_wav=str(ref))
    return Context(url="x", work_dir=tmp_path, config=cfg)


def write_translation(ctx, n):
    segs = [{"id": i, "start": float(i), "end": float(i) + 1.0,
             "text": f"en {i}", "text_pt": f"pt {i}"} for i in range(n)]
    write_json(ctx.path("04_translation.json"), {"segments": segs})


def install_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(synth_mod, "get_backend", lambda config: backend)
    return backend


def test_fresh_run_synthesizes_all_and_writes_manifest(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    write_translation(ctx, 4)
    backend = install_backend(monkeypatch)

    SynthesizeStage().run(ctx)

    assert backend.calls == ["pt 0", "pt 1", "pt 2", "pt 3"]
    assert valid_clips(ctx.path("05_clips.json"), ctx.path("05_clips")) is True
    manifest = read_json(ctx.path("05_clips.json"))
    assert manifest["sample_rate"] == SR
    assert [s["clip"] for s in manifest["segments"]] == [
        "0000.wav", "0001.wav", "0002.wav", "0003.wav"]
    assert all(abs(s["duration"] - 0.5) < 1e-6 for s in manifest["segments"])


def test_resume_only_synthesizes_missing_clips(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    write_translation(ctx, 4)

    # simulate a crash after clips 0 and 2 were written but before the manifest
    clips_dir = ctx.path("05_clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    for i in (0, 2):
        sf.write(str(clips_dir / f"{i:04d}.wav"),
                 np.zeros(SR // 2, dtype="float32"), SR)

    backend = install_backend(monkeypatch)
    SynthesizeStage().run(ctx)

    # only the missing clips hit the model; order in the manifest is still 0..3
    assert sorted(backend.calls) == ["pt 1", "pt 3"]
    manifest = read_json(ctx.path("05_clips.json"))
    assert [s["id"] for s in manifest["segments"]] == [0, 1, 2, 3]
    assert valid_clips(ctx.path("05_clips.json"), clips_dir) is True


def test_full_resume_does_not_load_model(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    write_translation(ctx, 3)

    # all clips already on disk, only the manifest is missing
    clips_dir = ctx.path("05_clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        sf.write(str(clips_dir / f"{i:04d}.wav"),
                 np.zeros(SR // 2, dtype="float32"), SR)

    def boom(config):
        raise AssertionError("TTS model must not load when nothing is pending")

    monkeypatch.setattr(synth_mod, "get_backend", boom)

    SynthesizeStage().run(ctx)

    manifest = read_json(ctx.path("05_clips.json"))
    assert manifest["sample_rate"] == SR  # recovered from an existing clip
    assert [s["id"] for s in manifest["segments"]] == [0, 1, 2]
    assert valid_clips(ctx.path("05_clips.json"), clips_dir) is True
