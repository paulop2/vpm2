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
    return all(
        "clip" in s and (clips_dir / s["clip"]).exists()
        for s in segs
    )