import pytest

from vpm2.url import ensure_single_video, sanitize_url


def test_sanitize_strips_shell_escaped_query():
    # What the shell passes for "watch\?v\=ID" inside double quotes.
    raw = r"https://www.youtube.com/watch\?v\=a2i9h2ip-nY"
    assert sanitize_url(raw) == "https://www.youtube.com/watch?v=a2i9h2ip-nY"


def test_sanitize_leaves_clean_url_untouched():
    url = "https://www.youtube.com/watch?v=a2i9h2ip-nY"
    assert sanitize_url(url) == url


def test_sanitize_trims_whitespace():
    assert sanitize_url("  https://youtu.be/abc \n") == "https://youtu.be/abc"


def test_ensure_single_video_accepts_video():
    ensure_single_video("https://youtu.be/abc", {"id": "abc", "title": "x"})


def test_ensure_single_video_rejects_playlist_type():
    with pytest.raises(SystemExit):
        ensure_single_video("url", {"_type": "playlist", "id": "UCxxxx"})


def test_ensure_single_video_rejects_channel_entries():
    with pytest.raises(SystemExit):
        ensure_single_video("url", {"id": "UCxxxx", "entries": [{"id": "v1"}]})
