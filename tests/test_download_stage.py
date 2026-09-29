import pytest

from vpm2.artifacts import read_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import download as download_mod
from vpm2.stages.download import DownloadStage


def make_ctx(tmp_path):
    return Context(url="https://youtu.be/abc", work_dir=tmp_path, config=Config())


class _FakeYDL:
    """Context manager standing in for yt_dlp.YoutubeDL."""

    def __init__(self, info=None, exc=None):
        self._info = info
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def extract_info(self, url, download=True):
        if self._exc is not None:
            raise self._exc
        return self._info


def test_download_error_becomes_systemexit_with_url(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    error = download_mod.yt_dlp.utils.DownloadError("Video unavailable")
    monkeypatch.setattr(
        download_mod.yt_dlp, "YoutubeDL", lambda opts: _FakeYDL(exc=error)
    )

    with pytest.raises(SystemExit) as ei:
        DownloadStage().run(ctx)

    message = str(ei.value)
    assert message.startswith("[vpm2]")
    assert ctx.url in message
    assert "Video unavailable" in message


def test_download_writes_meta(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    info = {"id": "abc", "title": "Título", "duration": 12.5}
    monkeypatch.setattr(
        download_mod.yt_dlp, "YoutubeDL", lambda opts: _FakeYDL(info=info)
    )

    DownloadStage().run(ctx)

    meta = read_json(ctx.path("01_meta.json"))
    assert meta == {
        "id": "abc",
        "title": "Título",
        "url": ctx.url,
        "duration": 12.5,
    }
