import json
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_segments(path: Path) -> list | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    segs = data.get("segments")
    if not isinstance(segs, list) or not segs:
        return None
    return segs


def valid_transcript(path: Path) -> bool:
    segs = _load_segments(path)
    if segs is None:
        return False
    return all(
        {"id", "start", "end", "text"} <= set(s) and str(s["text"]).strip()
        for s in segs
    )


def valid_translation(path: Path) -> bool:
    segs = _load_segments(path)
    if segs is None:
        return False
    return all(
        {"id", "start", "end", "text", "text_pt"} <= set(s)
        and str(s["text_pt"]).strip()
        for s in segs
    )


def valid_clips(path: Path, clips_dir: Path) -> bool:
    segs = _load_segments(path)
    if segs is None:
        return False
    for s in segs:
        if "clip" not in s or not (clips_dir / s["clip"]).exists():
            return False
        # `duration` is the synthesized PT clip length (spec Frente 3): the
        # assemble stage reads it, so a manifest without it (or with a
        # non-positive value) is a partial artifact and must not mark stage 05
        # as done.
        try:
            duration = float(s["duration"])
        except (KeyError, TypeError, ValueError):
            return False
        if not duration > 0:
            return False
    return True