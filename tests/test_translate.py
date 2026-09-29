import threading
import time

import pytest
import requests

from vpm2.artifacts import read_json, write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import translate as translate_mod
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


class _FakeResponse:
    def __init__(self, status_code, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data if json_data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Server Error", response=self
            )

    def json(self):
        return self._json


def _patch_post(monkeypatch, fn):
    monkeypatch.setattr(translate_mod.requests, "post", fn)


def test_connection_error_suggests_ollama_serve(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    _write_transcript(ctx, 1)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    _patch_post(monkeypatch, boom)

    with pytest.raises(SystemExit) as ei:
        TranslateStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert ctx.config.ollama_url in message
    assert "ollama serve" in message


def test_timeout_suggests_ollama_serve(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    _write_transcript(ctx, 1)

    def boom(*args, **kwargs):
        raise requests.Timeout("timed out")

    _patch_post(monkeypatch, boom)

    with pytest.raises(SystemExit) as ei:
        TranslateStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert ctx.config.ollama_url in message
    assert "ollama serve" in message


def test_http_404_suggests_ollama_pull(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    _write_transcript(ctx, 1)
    _patch_post(
        monkeypatch,
        lambda *a, **k: _FakeResponse(404, '{"error":"model not found"}'),
    )

    with pytest.raises(SystemExit) as ei:
        TranslateStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert "ollama pull" in message
    assert ctx.config.ollama_model in message


def test_other_http_error_reports_status_and_truncated_body(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    _write_transcript(ctx, 1)
    body = "erro interno " * 50
    _patch_post(monkeypatch, lambda *a, **k: _FakeResponse(500, body))

    with pytest.raises(SystemExit) as ei:
        TranslateStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert "500" in message
    assert "erro interno" in message
    # body is summarized, not dumped whole
    assert len(message) < len(body)

