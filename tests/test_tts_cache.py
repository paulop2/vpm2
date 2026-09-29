from pathlib import Path

from vpm2.tts_cache import clear_tts_cache


def test_clear_removes_wav_files_and_counts(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"a")
    (tmp_path / "b.wav").write_bytes(b"b")
    assert clear_tts_cache(tmp_path) == 2
    assert list(tmp_path.iterdir()) == []


def test_clear_removes_interrupted_tmp_files(tmp_path):
    # Synthesize writes `.<name>.wav.tmp` and renames atomically; an
    # interrupted run leaves a dot-prefixed `*.tmp` behind.
    (tmp_path / ".abc.wav.tmp").write_bytes(b"junk")
    (tmp_path / ".abc.tmp").write_bytes(b"junk")
    assert clear_tts_cache(tmp_path) == 2
    assert list(tmp_path.iterdir()) == []


def test_clear_keeps_non_wav_and_subdirs(tmp_path):
    (tmp_path / "keep.wav").write_bytes(b"wav")
    (tmp_path / "keep.txt").write_bytes(b"txt")
    (tmp_path / "keep.json").write_bytes(b"{}")
    (tmp_path / "keep.tmp").write_bytes(b"tmp-but-not-dot-prefixed")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.wav").write_bytes(b"nested")

    assert clear_tts_cache(tmp_path) == 1
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"keep.txt", "keep.json", "keep.tmp", "sub"}
    assert (sub / "nested.wav").exists()


def test_clear_missing_dir_returns_zero(tmp_path):
    missing = tmp_path / "nao-existe"
    assert clear_tts_cache(missing) == 0


def test_clear_accepts_str_path(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"a")
    assert clear_tts_cache(str(tmp_path)) == 1


def test_clear_empty_dir_returns_zero(tmp_path):
    assert clear_tts_cache(tmp_path) == 0


def test_clear_returns_path_agnostic(tmp_path):
    # Path | str both accepted; ensure Path instances are not required.
    assert clear_tts_cache(Path(tmp_path)) == 0
