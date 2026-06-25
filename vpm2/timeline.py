from dataclasses import dataclass


@dataclass
class PlacedClip:
    id: int
    start: float
    speed: float
    src_duration: float
    out_duration: float


def plan_timeline(segments, video_duration, max_speed=1.25, allow_push=True):
    segs = sorted(segments, key=lambda s: s["id"])
    placed: list[PlacedClip] = []
    cursor = 0.0

    for i, s in enumerate(segs):
        target_start = max(float(s["start"]), cursor)
        if i + 1 < len(segs):
            boundary = float(segs[i + 1]["start"])
        else:
            boundary = float(video_duration)
        available = max(boundary - target_start, 0.0)

        src = float(s["duration"])
        if available > 0 and src > available:
            speed = min(src / available, max_speed)
        else:
            speed = 1.0
        out = src / speed

        placed.append(PlacedClip(
            id=int(s["id"]), start=target_start, speed=speed,
            src_duration=src, out_duration=out,
        ))

        # With push, the next clip cannot start before this one ends.
        # Without push, reset the cursor so the next clip is free to start at
        # its own original start (max(s["start"], 0.0) == s["start"]).
        cursor = (target_start + out) if allow_push else 0.0

    return placed