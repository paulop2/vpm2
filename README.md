# VPM2 — Vídeos Para Minha Mãe

Translate English YouTube videos into a synced PT-BR dub, fully local.

## Architecture

A staged pipeline. Each stage reads/writes artifacts in `work/<video-id>/`, is
resumable (skips if its output already exists and validates), and can be swapped
without touching the orchestrator. Heavy models (ASR → LLM → TTS) load on demand and
run **sequentially**, so GPU memory is freed between stages.

```
                            YouTube URL
                                 │
                                 ▼
  ┌──────────────┐   01_video.mp4   ┌──────────────────┐   02_audio.wav (16k mono)
  │  download    │ ───────────────► │  extract_audio   │ ──────────────┐
  │  (yt-dlp)    │   01_meta.json   │  (ffmpeg)        │               │
  └──────────────┘                  └──────────────────┘               │
                                                                       ▼
  ┌──────────────────┐  04_translation.json  ┌──────────────────┐  03_transcript.json
  │  translate       │ ◄──────────────────── │  transcribe      │ ◄───────────┘
  │  (Ollama LLM)    │ ──────────┐            │  (faster-whisper)│
  └──────────────────┘           │           └──────────────────┘
                                 ▼
  ┌──────────────────┐  05_clips/*.wav   ┌──────────────────┐  06_final.mp4
  │  synthesize      │ ────────────────► │  assemble        │ ──────────► output
  │  (Chatterbox TTS)│  05_clips.json    │  (timeline sync  │  06_audio_pt.wav
  │  + voice cloning │  ref_voice.wav    │   + ffmpeg mux)  │
  └──────────────────┘                   └──────────────────┘
```

| Stage | Tool | Input → Output |
|---|---|---|
| `download` | yt-dlp | URL → `01_video.mp4`, `01_meta.json` |
| `extract_audio` | ffmpeg | `01_video.mp4` → `02_audio.wav` (16 kHz mono) |
| `transcribe` | faster-whisper (GPU) | `02_audio.wav` → `03_transcript.json` (EN segments) |
| `translate` | Ollama LLM | `03_transcript.json` → `04_translation.json` (PT-BR) |
| `synthesize` | Chatterbox TTS (GPU) | `04_translation.json` → `05_clips/*.wav` (+ `ref_voice.wav`) |
| `assemble` | timeline sync + ffmpeg | clips + video → `06_final.mp4` |

The `assemble` stage uses a pure timeline algorithm (`vpm2/timeline.py`) to place each
clip, accelerating up to `max_speed` (default 1.25x) and pushing later clips when a
segment overruns its gap.

## Setup (WSL2 + NVIDIA GPU)

1. Install ffmpeg: `sudo apt install ffmpeg`
2. Install Ollama and pull a translation model: `ollama pull qwen3:8b`
3. Install deps: `uv sync`
4. Install a CUDA 12.8 PyTorch build (see Task: TTS).
5. (Optional) Pre-download the ASR + TTS weights so the first run doesn't stall —
   see "Model downloads & caching" below.

## Model downloads & caching

Three models power the pipeline, downloaded from **different** places:

| Model | Stage | Source | Size | When it downloads |
|---|---|---|---|---|
| faster-whisper `large-v3` | `transcribe` | HuggingFace Hub | ~2.9 GB | first run |
| Chatterbox Multilingual | `synthesize` | HuggingFace Hub | ~2–3 GB | first run |
| `qwen3:8b` (translation LLM) | `translate` | Ollama (`ollama pull`) | ~5 GB | `ollama pull` |

The HuggingFace weights (Whisper, Chatterbox) are **downloaded once** and cached in
`~/.cache/huggingface/hub`. They are **not** re-downloaded on later runs — every run
after the first loads them from disk. Only the very first video pays this one-time
cost (it happens lazily when each stage first loads its model, not during `uv sync`).

To pre-warm the cache during setup (so the first real run is fast and the system is
fully offline-ready), trigger the downloads ahead of time:

```bash
# ASR (faster-whisper large-v3)
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16')"

# TTS (Chatterbox Multilingual)
uv run python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cuda')"
```

The translation model is separate and lives in Ollama's own store — pull it (and any
alternative you want to compare) with `ollama pull <model>`.

## Usage

```bash
uv run vpm2 "https://www.youtube.com/watch?v=..."
```

Output: `work/<video-id>/06_final.mp4`.

Swap the translation model with `--ollama-model` (e.g. a dedicated translator):

```bash
uv run vpm2 "<url>" --ollama-model zongwei/gemma3-translator
```

## Smoke test

See `tests/` for unit tests (`uv run pytest`). End-to-end requires GPU + network.

## End-to-end smoke test (requires GPU + network)

1. Start Ollama and pull the model: `ollama pull qwen3:8b`
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
8. Compare translation models: re-run with a different model and force the translate
   stage, e.g. `uv run vpm2 "<url>" --ollama-model zongwei/gemma3-translator --force translate`.
