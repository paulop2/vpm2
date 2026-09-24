from vpm2.asr.base import RawSegment


def normalize_segments(raw: list[RawSegment]) -> list[dict]:
    cleaned = [
        RawSegment(float(s.start), float(s.end), s.text.strip())
        for s in raw
    ]
    cleaned = [s for s in cleaned if s.text]
    cleaned.sort(key=lambda s: (s.start, s.end))
    return [
        {"id": i, "start": s.start, "end": s.end, "text": s.text}
        for i, s in enumerate(cleaned)
    ]
