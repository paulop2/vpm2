import numpy as np
import soundfile as sf

from vpm2.artifacts import write_json
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
        return np.zeros(SR // 2, dtype="float32")


def _ctx(tmp_path, name, cache_dir):
    ref = tmp_path / "ref.wav"
    if not ref.exists():
        sf.write(str(ref), np.zeros(SR, dtype="float32"), SR)
    cfg = Config(voice_mode="preset", preset_ref_wav=str(ref),
                 tts_cache=True, tts_cache_dir=str(cache_dir))
    ctx = Context(url="x", work_dir=tmp_path / name, config=cfg)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(ctx.path("04_translation.json"), {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "en", "text_pt": "pt"}]})
    return ctx


def test_cache_reused_across_videos(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    fake = FakeBackend()
    monkeypatch.setattr(synth_mod, "get_backend", lambda cfg: fake)

    SynthesizeStage().run(_ctx(tmp_path, "v1", cache))
    SynthesizeStage().run(_ctx(tmp_path, "v2", cache))

    assert fake.calls == ["pt"]
