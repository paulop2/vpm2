from unittest.mock import patch

import pytest

from vpm2 import cli
from vpm2.config import Config


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
    with _patch_ydl(info=info), pytest.raises(SystemExit):
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


def test_main_rejects_unknown_force_stage():
    # argparse choices must fail at parse time (exit 2), before any network.
    with pytest.raises(SystemExit) as ei:
        cli.main(["http://x", "--force", "nao-existe"])
    assert ei.value.code == 2


def _run_main(argv):
    """Run cli.main with heavy work stubbed; return the Config it built."""
    captured = {}
    with patch.object(cli, "check_ffmpeg"), patch.object(cli, "check_ollama"), \
         patch.object(cli, "_resolve_id", return_value="vid"), \
         patch.object(cli, "run_pipeline",
                      side_effect=lambda ctx, **k: captured.update(config=ctx.config)):
        rc = cli.main(argv)
    assert rc == 0
    return captured["config"]


def test_main_passes_voices_dir_to_config():
    cfg = _run_main(["http://x", "--voices-dir", "minhas-vozes"])
    assert cfg.voices_dir == "minhas-vozes"


def test_main_passes_tts_cache_dir_to_config():
    cfg = _run_main(["http://x", "--tts-cache-dir", "meu-cache"])
    assert cfg.tts_cache_dir == "meu-cache"


def test_main_no_tts_cache_disables_cache():
    cfg = _run_main(["http://x", "--no-tts-cache"])
    assert cfg.tts_cache is False


def test_main_keeps_config_defaults_without_new_flags():
    default = Config()
    cfg = _run_main(["http://x"])
    assert cfg.voices_dir == default.voices_dir
    assert cfg.tts_cache_dir == default.tts_cache_dir
    assert cfg.tts_cache == default.tts_cache


def test_main_without_url_errors_with_code_2():
    # With `url` optional, a bare invocation must still fail arg parsing.
    with pytest.raises(SystemExit) as ei:
        cli.main([])
    assert ei.value.code == 2


def test_main_clear_tts_cache_returns_0_without_heavy_work(tmp_path):
    (tmp_path / "clip.wav").write_bytes(b"clip")

    def _boom(*a, **k):  # pragma: no cover - only runs on regression
        raise AssertionError("clear-tts-cache must not do heavy work")

    with patch.object(cli, "check_ffmpeg", _boom), \
         patch.object(cli, "check_ollama", _boom), \
         patch.object(cli, "run_pipeline", _boom):
        rc = cli.main(["--clear-tts-cache", "--tts-cache-dir", str(tmp_path)])

    assert rc == 0
    assert list(tmp_path.iterdir()) == []


def test_main_clear_tts_cache_works_without_url(tmp_path, capsys):
    (tmp_path / "clip.wav").write_bytes(b"clip")
    rc = cli.main(["--clear-tts-cache", "--tts-cache-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[vpm2] cache de TTS limpo: 1 arquivo(s) removido(s) de" in out
