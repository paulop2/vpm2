import pytest

from vpm2.voices import profile as vp


def _write_bytes(path, content=b"RIFFfake"):
    path.write_bytes(content)
    return path


def test_make_profile_captures_sha_and_span(tmp_path):
    ref = _write_bytes(tmp_path / "ref.wav", b"abc")
    p = vp.make_profile("bob", ref, source_video="vid1", span=(1.0, 11.0),
                        transcript="hi", backend="chatterbox")
    assert p.id == "bob"
    assert p.span_start == 1.0 and p.span_end == 11.0
    assert p.sha256 == vp.sha256_file(ref)


def test_save_and_load_roundtrip(tmp_path):
    ref = _write_bytes(tmp_path / "ref.wav")
    p = vp.make_profile("bob", ref, source_video="v", span=(0.0, 7.0),
                        transcript="t", backend="chatterbox")
    vp.save_profile(tmp_path, p)
    assert vp.load_profile(tmp_path, "bob") == p


def test_resolve_reference_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        vp.resolve_reference(tmp_path, "nope")


def test_cache_key_stable_and_sensitive():
    a = vp.cache_key("sha1", "hello", {"backend": "chatterbox"})
    b = vp.cache_key("sha1", "hello", {"backend": "chatterbox"})
    c = vp.cache_key("sha1", "hello", {"backend": "other"})
    d = vp.cache_key("sha1", "HELLO", {"backend": "chatterbox"})
    assert a == b
    assert a != c
    assert a != d


def test_transcript_for_window_joins_overlapping_segments():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "one"},
        {"start": 2.0, "end": 4.0, "text": "two"},
        {"start": 10.0, "end": 12.0, "text": "far"},
    ]
    assert vp.transcript_for_window(segs, (1.0, 3.0)) == "one two"
