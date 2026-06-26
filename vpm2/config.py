from dataclasses import dataclass


@dataclass
class Config:
    asr_model: str = "large-v3"
    ollama_model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    # Segments translate independently (context comes from the source transcript,
    # not from prior outputs), so we fan out concurrent requests to Ollama. They
    # share the one loaded model server-side -- only KV-cache grows per slot, so
    # this costs no extra resident VRAM. Keep modest to bound that KV-cache.
    translate_workers: int = 8
    tts_backend: str = "chatterbox"
    voice_mode: str = "cloning"         # "cloning" | "preset"
    preset_ref_wav: str | None = None   # required when voice_mode == "preset"
    source_lang: str = "en"
    target_lang: str = "pt"
    # PT-BR dubs run noticeably longer than the English source, so allow up to
    # 1.5x time-stretch (pitch preserved by ffmpeg atempo) before clips spill
    # past the video. Only clips that overrun are sped up.
    max_speed: float = 1.5
    allow_push: bool = True
    # A dub should "just play" PT-BR in every player. Keeping the English track
    # as a second stream makes many players fall back to it, so default to a
    # single PT-BR track; opt back in with --keep-original-audio.
    keep_original_audio: bool = False
