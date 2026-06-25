from unittest.mock import patch

import pytest

from vpm2 import cli


class FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL used as a context manager."""

    def __init__(self, info=None, exc=None):
        self._info = info
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        if self._exc is not None:
            raise self._exc
        return self._info


def _patch_ydl(info=None, exc=None):
    return patch.object(cli.yt_dlp, "YoutubeDL", lambda *a, **k: FakeYDL(info, exc))


def test_resolve_id_returns_video_id():
    with _patch_ydl(info={"id": "a2i9h2ip-nY"}):
        assert cli._resolve_id("https://youtu.be/a2i9h2ip-nY") == "a2i9h2ip-nY"


def test_resolve_id_rejects_channel():
    info = {"_type": "playlist", "id": "UCxxxx", "entries": [{"id": "v1"}]}
    with _patch_ydl(info=info):
        with pytest.raises(SystemExit):
            cli._resolve_id("https://www.youtube.com/@SomeChannel")


def test_resolve_id_falls_back_to_slug_on_extract_error():
    # A failed extraction must not crash -- it degrades to a filesystem-safe slug.
    with _patch_ydl(exc=RuntimeError("network down")):
        slug = cli._resolve_id("https://youtu.be/a2i9h2ip-nY")
    assert slug and "/" not in slug


def test_main_sanitizes_escaped_url_before_use():
    captured = {}
    with patch.object(cli, "check_ffmpeg"), patch.object(cli, "check_ollama"), \
         patch.object(cli, "_resolve_id", return_value="vid"), \
         patch.object(cli, "run_pipeline",
                      side_effect=lambda ctx, **k: captured.update(url=ctx.url)):
        rc = cli.main([r"https://www.youtube.com/watch\?v\=a2i9h2ip-nY"])
    assert rc == 0
    assert captured["url"] == "https://www.youtube.com/watch?v=a2i9h2ip-nY"
