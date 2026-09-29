import pytest

from vpm2.artifacts import read_json
from vpm2.asr.base import RawSegment
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import transcribe as transcribe_mod
from vpm2.stages.transcribe import TranscribeStage


class FakeASRBackend:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path):
        return self._segments


class RaisingASRBackend:
    def __init__(self, exc):
        self._exc = exc

    def transcribe(self, audio_path):
        raise self._exc


def make_ctx(tmp_path):
    return Context(url="x", work_dir=tmp_path, config=Config())


def test_stage_writes_normalized_transcript(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    fake = FakeASRBackend([
        RawSegment(2.0, 3.0, "b"),
        RawSegment(0.0, 1.0, "a"),
        RawSegment(1.0, 2.0, "   "),
    ])
    monkeypatch.setattr(transcribe_mod, "get_asr_backend", lambda cfg: fake)

    TranscribeStage().run(ctx)

    data = read_json(tmp_path / "03_transcript.json")
    assert data["language"] == "en"
    assert [s["text"] for s in data["segments"]] == ["a", "b"]
    assert [s["id"] for s in data["segments"]] == [0, 1]


def test_is_done_false_when_missing(tmp_path):
    assert TranscribeStage().is_done(make_ctx(tmp_path)) is False


def test_is_done_true_after_run(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: FakeASRBackend([RawSegment(0.0, 1.0, "a")]),
    )
    stage = TranscribeStage()
    stage.run(ctx)
    assert stage.is_done(ctx) is True


def test_runtime_error_cuda_out_of_memory_suggests_whisper(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: RaisingASRBackend(exc),
    )

    with pytest.raises(SystemExit) as ei:
        TranscribeStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert "whisper" in message
    assert ctx.config.asr_backend in message


def test_runtime_error_cuda_uppercase_is_matched(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    exc = RuntimeError("CUDA error: no kernel image is available")
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: RaisingASRBackend(exc),
    )

    with pytest.raises(SystemExit) as ei:
        TranscribeStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert "whisper" in message
    assert ctx.config.asr_backend in message


def test_import_error_from_factory_suggests_whisper(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)

    def raising_factory(cfg):
        raise ImportError("No module named 'nemo'")

    monkeypatch.setattr(transcribe_mod, "get_asr_backend", raising_factory)

    with pytest.raises(SystemExit) as ei:
        TranscribeStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert "whisper" in message
    assert ctx.config.asr_backend in message


def test_import_error_from_backend_use_suggests_whisper(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: RaisingASRBackend(ImportError("No module named 'nemo'")),
    )

    with pytest.raises(SystemExit) as ei:
        TranscribeStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert "whisper" in message


def test_other_value_error_propagates(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: RaisingASRBackend(ValueError("boom")),
    )

    with pytest.raises(ValueError):
        TranscribeStage().run(ctx)


def test_non_cuda_runtime_error_propagates(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: RaisingASRBackend(RuntimeError("model file is corrupt")),
    )

    with pytest.raises(RuntimeError):
        TranscribeStage().run(ctx)
