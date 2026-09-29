from pathlib import Path


def clear_tts_cache(cache_dir: Path | str) -> int:
    """Remove cached TTS clips from ``cache_dir`` and return how many files went.

    Deletes ``*.wav`` clips plus the dot-prefixed ``.*.tmp`` files left behind
    by an interrupted atomic write (see ``stages.synthesize._write_wav_atomic``).
    Subdirectories and any other file type are left untouched. A missing
    directory is a no-op that returns 0.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(".wav") or (name.startswith(".") and name.endswith(".tmp")):
            entry.unlink()
            removed += 1
    return removed
