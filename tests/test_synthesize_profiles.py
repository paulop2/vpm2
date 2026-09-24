import numpy as np
import pytest
import soundfile as sf

from vpm2.artifacts import write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import synthesize as synth_mod
from vpm2.stages.synthesize import SynthesizeStage
from vpm2.voices import profile as vp

SR = 24000


class FakeBackend:
    sample_rate = SR

    def __init__(self):
        self.refs = []

    def synth(self, text, ref):
        self.refs.append(ref)
        return np.zeros(SR // 2, dtype="float32")


def _ctx(tmp_path, **cfg):
    ctx = Context(url="vid", work_dir=tmp_path / "work", config=Config(**cfg))
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(ctx.path("04_translation.json"), {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "en", "text_pt": "pt"}]})
    return ctx


def test_profile_mode_uses_saved_reference(tmp_path, monkeypatch):
    voices = tmp_path / "voices"
    ref = voices / "bob" / "ref.wav"
    ref.parent.mkdir(parents=True)
    sf.write(str(ref), np.zeros(SR, dtype="float32"), SR)
    p = vp.make_profile("bob", ref, source_video="v", span=(0.0, 1.0),
                        transcript="hi", backend="chatterbox")
    vp.save_profile(voices, p)

    ctx = _ctx(tmp_path, voice_mode="profile", voice_profile="bob",
               voices_dir=str(voices), tts_cache_dir=str(tmp_path / "cache"))
    fake = FakeBackend()
    monkeypatch.setattr(synth_mod, "get_backend", lambda cfg: fake)

    SynthesizeStage().run(ctx)
    assert fake.refs == [ref]


def test_profile_mode_without_id_raises(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, voice_mode="profile",
               tts_cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(synth_mod, "get_backend", lambda cfg: FakeBackend())
    with pytest.raises(ValueError):
        SynthesizeStage().run(ctx)
