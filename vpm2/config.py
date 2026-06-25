from dataclasses import dataclass


@dataclass
class Config:
    asr_model: str = "large-v3"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    tts_backend: str = "chatterbox"
    voice_mode: str = "cloning"         # "cloning" | "preset"
    preset_ref_wav: str | None = None   # required when voice_mode == "preset"
    source_lang: str = "en"
    target_lang: str = "pt"
    max_speed: float = 1.25
    allow_push: bool = True
    keep_original_audio: bool = True
