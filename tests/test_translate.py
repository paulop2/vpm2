import threading
import time

from vpm2.artifacts import read_json, write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages.translate import TranslateStage


def make_ctx(tmp_path, **cfg):
    return Context(url="x", work_dir=tmp_path, config=Config(**cfg))


def _write_transcript(ctx, n):
    segs = [{"id": i, "start": float(i), "end": float(i) + 1.0,
             "text": f"line {i}"} for i in range(n)]
    write_json(ctx.path("03_transcript.json"), {"segments": segs})


def test_translate_preserves_order_and_context(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, translate_workers=4)
    _write_transcript(ctx, 5)

    seen = {}

    def fake(self, ctx, text, prev, nxt):
        seen[text] = (prev, nxt)
        return f"PT:{text}"

    monkeypatch.setattr(TranslateStage, "_translate_one", fake)
    TranslateStage().run(ctx)

    out = read_json(ctx.path("04_translation.json"))["segments"]
    # order matches the source transcript despite out-of-order completion
    assert [s["id"] for s in out] == [0, 1, 2, 3, 4]
    assert [s["text_pt"] for s in out] == [f"PT:line {i}" for i in range(5)]
    # neighbor context is taken from the source transcript, edges are None
    assert seen["line 0"] == (None, "line 1")
    assert seen["line 2"] == ("line 1", "line 3")
    assert seen["line 4"] == ("line 3", None)


def test_translate_actually_runs_concurrently(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path, translate_workers=5)
    _write_transcript(ctx, 5)

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake(self, ctx, text, prev, nxt):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return f"PT:{text}"

    monkeypatch.setattr(TranslateStage, "_translate_one", fake)
    TranslateStage().run(ctx)

    # with 5 workers and 5 segments, more than one should overlap
    assert peak > 1
