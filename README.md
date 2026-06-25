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

## End-to-end smoke test (requires GPU + network)

1. Start Ollama and pull the model: `ollama pull qwen2.5:7b-instruct`
2. Pick a short (~30–60s) English clip URL.
3. Cloning voice (default, zero-config — reference auto-extracted from the video):
   `uv run vpm2 "<url>"`
4. Preset voice (provide a clean PT reference wav to compare):
   `uv run vpm2 "<url>" --voice-mode preset --preset-ref ref_pt.wav`
5. Inspect artifacts in `work/<id>/`: open `03_transcript.json`,
   `04_translation.json`, listen to `05_clips/*.wav` and `ref_voice.wav`, then play
   `06_final.mp4` (the muted second track is the original English audio).
6. Re-run the same command — every stage should print "skipping (done)".
7. Force a re-translate: `uv run vpm2 "<url>" --force translate`.
