# VPM2 — Vídeos Para Minha Mãe

Translate English YouTube videos into a synced PT-BR dub, fully local.

## Setup (WSL2 + NVIDIA GPU)

1. Install ffmpeg: `sudo apt install ffmpeg`
2. Install Ollama and pull a translation model: `ollama pull qwen2.5:7b-instruct`
3. Install deps: `uv sync`
4. Install a CUDA 12.8 PyTorch build (see Task: TTS).

## Usage

```bash
uv run vpm2 "https://www.youtube.com/watch?v=..."
```

Output: `work/<video-id>/06_final.mp4`.

## Smoke test

See `tests/` for unit tests (`uv run pytest`). End-to-end requires GPU + network.
