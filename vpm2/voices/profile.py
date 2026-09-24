import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    source_video: str
    span_start: float
    span_end: float
    transcript: str
    sha256: str
    created_at: str
    backend: str


def profile_dir(voices_dir: Path | str, profile_id: str) -> Path:
    return Path(voices_dir) / profile_id


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_profile(profile_id, ref_wav, *, source_video, span, transcript, backend):
    return VoiceProfile(
        id=profile_id,
        source_video=source_video,
        span_start=float(span[0]),
        span_end=float(span[1]),
        transcript=transcript,
        sha256=sha256_file(ref_wav),
        created_at=datetime.now(timezone.utc).isoformat(),
        backend=backend,
    )


def save_profile(voices_dir, profile: VoiceProfile) -> None:
    d = profile_dir(voices_dir, profile.id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.json").write_text(
        json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8",
    )


def load_profile(voices_dir, profile_id: str) -> VoiceProfile:
    path = profile_dir(voices_dir, profile_id) / "profile.json"
    return VoiceProfile(**json.loads(path.read_text(encoding="utf-8")))


def resolve_reference(voices_dir, profile_id: str) -> Path:
    ref = profile_dir(voices_dir, profile_id) / "ref.wav"
    if not ref.exists():
        raise FileNotFoundError(f"voice profile '{profile_id}' has no ref.wav")
    return ref


def cache_key(profile_sha: str, text: str, params: dict | None = None) -> str:
    payload = json.dumps(
        {"profile_sha": profile_sha, "text": text, "params": params or {}},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def transcript_for_window(segments: list[dict], window: tuple[float, float]) -> str:
    start, end = window
    parts = [
        s["text"] for s in segments
        if float(s["end"]) > start and float(s["start"]) < end
    ]
    return " ".join(parts).strip()
