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
  ┌──────────────────┐  04_translation.json  ┌────────────────────┐  03_transcript.json
  │  translate       │ ◄──────────────────── │  transcribe        │ ◄───────────┘
  │  (Ollama LLM)    │ ──────────┐           │  (Parakeet/Whisper)│
  └──────────────────┘           │           └────────────────────┘
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
| `transcribe` | **Parakeet TDT 0.6B v2 (NeMo, GPU; default)** or faster-whisper `large-v3` (`--asr-backend whisper`) | `02_audio.wav` → `03_transcript.json` (EN segments) |
| `translate` | Ollama LLM (`qwen3:8b`, `--translate-workers` 8) | `03_transcript.json` → `04_translation.json` (PT-BR) |
| `synthesize` | Chatterbox TTS (GPU) + voice cloning | `04_translation.json` → `05_clips/*.wav`, `05_clips.json` (+ `ref_voice.wav` with `--voice-mode cloning`) |
| `assemble` | timeline sync + ffmpeg | clips + video → `06_final.mp4` (+ `06_audio_pt.wav`) |

The `assemble` stage uses a pure timeline algorithm (`vpm2/timeline.py`) to place each
clip, accelerating up to `max_speed` (default 1.5x) and pushing later clips when a
segment overruns its gap.

## Setup (WSL2 + NVIDIA GPU)

1. Install ffmpeg: `sudo apt install ffmpeg`
2. Install Ollama and pull a translation model: `ollama pull qwen3:8b`
3. Install the full GPU stack (Parakeet/NeMo, faster-whisper, Chatterbox TTS):
   `uv sync --extra gpu`.
   The extra pulls `torch` from the CUDA 12.8 index configured in
   `[tool.uv.sources]`; plain `uv sync` installs only the lightweight deps needed
   for the test suite.
4. Configure HuggingFace access — see ["Configuration"](#configuration) below
   (`HF_TOKEN`).
5. (Optional) Pre-download the ASR + TTS weights so the first run doesn't stall —
   see ["Model downloads & caching"](#model-downloads--caching) below.

## Configuration

`vpm2` loads a `.env` file from the working directory at startup (via
`python-dotenv`, in `vpm2/cli.py`). Create `.env` in the repo root to provide:

```bash
# HuggingFace token, used by huggingface_hub when the transcribe stage downloads
# the Parakeet / Whisper / Chatterbox weights. Required for gated repos and
# recommended to avoid anonymous rate limits.
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The token is passed to the model-loading libraries through the process
environment; it never has to be committed (`.env` is git-ignored).

## Model downloads & caching

Four models power the pipeline, downloaded from **different** places:

| Model | Stage | Source | Size / params | When it downloads |
|---|---|---|---|---|
| Parakeet TDT `nvidia/parakeet-tdt-0.6b-v2` (default) | `transcribe` | HuggingFace Hub (NeMo) | ~0.6 B params | first run |
| faster-whisper `large-v3` (`--asr-backend whisper`) | `transcribe` | HuggingFace Hub | ~2.9 GB | first run |
| Chatterbox Multilingual | `synthesize` | HuggingFace Hub | ~2–3 GB | first run |
| `qwen3:8b` (translation LLM) | `translate` | Ollama (`ollama pull`) | ~5 GB | `ollama pull` |

The HuggingFace weights (Parakeet, Whisper, Chatterbox) are **downloaded once** and
cached in `~/.cache/huggingface/hub`. They are **not** re-downloaded on later runs —
every run after the first loads them from disk. Only the very first video pays this
one-time cost (it happens lazily when each stage first loads its model, not during
`uv sync`).

To pre-warm the cache during setup (so the first real run is fast and the system is
fully offline-ready), trigger the downloads ahead of time:

```bash
# ASR (Parakeet TDT 0.6B v2, default) -- loads NeMo from the HF cache
uv run python scripts/check_parakeet.py <some-16k-mono.wav>

# ASR fallback (faster-whisper large-v3)
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

Every flag accepted by `python -m vpm2.cli --help` is listed below.

| Flag | Default | Description |
|---|---|---|
| `url` | — | YouTube video URL. Optional only with `--clear-tts-cache`. |
| `-h`, `--help` | — | Show this help message and exit. |
| `--voice-mode {cloning,preset,profile}` | `cloning` | Reference-voice strategy for TTS (see ["Voices"](#voices)). |
| `--voice-profile ID` | — | Saved voice profile id; required with `--voice-mode profile`. |
| `--save-profile ID` | — | With `cloning`, persist the auto-extracted reference under this profile id. |
| `--voices-dir DIR` | `voices` | Directory where voice profiles are stored. |
| `--asr-backend {parakeet,whisper}` | `parakeet` | ASR engine for the `transcribe` stage. |
| `--preset-ref FILE.wav` | — | Clean PT reference wav; required with `--voice-mode preset`. |
| `--force STAGE` | — | Re-run the pipeline starting at `STAGE` (`download`, `extract_audio`, `transcribe`, `translate`, `synthesize`, `assemble`). |
| `--ollama-model NAME` | `qwen3:8b` | Translation LLM served by Ollama. |
| `--translate-workers N` | `8` | Concurrent translation requests to Ollama. |
| `--asr-model NAME` | `large-v3` | Whisper model id (used with `--asr-backend whisper`). |
| `--work-root DIR` | `work` | Root directory for per-video artifacts. |
| `--tts-cache-dir DIR` | `work/_tts_cache` | Directory of the cross-video TTS clip cache. |
| `--no-tts-cache` | off | Disable the TTS cache for this run. |
| `--clear-tts-cache` | off | Delete cached TTS clips and exit (does not run the pipeline). |
| `--keep-original-audio` | off | Keep the English audio as a second track (off → output has only the PT-BR dub). |
| `--max-speed F` | `1.5` | Max time-stretch for clips that overrun their slot (pitch preserved; ffmpeg `atempo` caps at 2.0). |

Examples:

```bash
# Swap the translation model (e.g. a dedicated translator)
uv run vpm2 "<url>" --ollama-model zongwei/gemma3-translator

# Re-transcribe with faster-whisper instead of Parakeet
uv run vpm2 "<url>" --asr-backend whisper --force transcribe

# Keep the original English audio as a second track
uv run vpm2 "<url>" --keep-original-audio
```

## Voices

`--voice-mode` selects how the TTS reference is obtained (default `cloning`):

- **`cloning`** (default, zero-config): the reference is auto-extracted from the
  video's own audio (VAD window) and saved to `work/<id>/ref_voice.wav`.
  Add `--save-profile <id>` to promote that reference into a reusable profile.
- **`preset`**: use a clean PT reference wav you provide:
  `--voice-mode preset --preset-ref ref_pt.wav`.
- **`profile`**: reuse a reference saved earlier:
  `--voice-mode profile --voice-profile <id>`.

**Profiles** live under `--voices-dir` (default `voices/`), one directory per id:

```
voices/<id>/
  ref.wav        # reference clip (ideally 6-10s of clean speech)
  profile.json   # {id, source_video, span_start, span_end, transcript, sha256, created_at, backend}
```

Creating a profile:

```bash
# Extract the reference from the video and persist it under voices/bob/
uv run vpm2 "<url>" --save-profile bob

# Reuse it on every later video without re-extracting
uv run vpm2 "<other-url>" --voice-mode profile --voice-profile bob
```

To improve a clone over time, replace `voices/<id>/ref.wav` (there is no training
in v1 — curation is substitution).

**TTS cache.** Synthesized clips are cached across videos in `--tts-cache-dir`
(default `work/_tts_cache`), keyed by `hash(profile, text, params)`, so repeated
lines are never re-synthesized. The cache is invalidated automatically when the
reference or the parameters change. Clear it explicitly with:

```bash
# Wipe the cache and exit (no URL, no pipeline)
uv run vpm2 --clear-tts-cache

# ...or point at a custom cache directory
uv run vpm2 --clear-tts-cache --tts-cache-dir work/_tts_cache
```

Disable the cache for a one-off run with `--no-tts-cache`.

## Smoke test ponta a ponta (GPU + rede)

Unit tests are lightweight and GPU-free: `uv run pytest`. The end-to-end flow below
requires a GPU and network.

1. Start Ollama and pull the model: `ollama pull qwen3:8b`
2. Pick a short (~30–60s) English clip URL.
3. Cloning voice (default, zero-config — reference auto-extracted from the video):
   `uv run vpm2 "<url>"`
4. Preset voice (provide a clean PT reference wav to compare):
   `uv run vpm2 "<url>" --voice-mode preset --preset-ref ref_pt.wav`
5. Inspect artifacts in `work/<id>/`: open `03_transcript.json`,
   `04_translation.json`, listen to `05_clips/*.wav` and `ref_voice.wav`, then play
   `06_final.mp4` (by default it has a single PT-BR track; the original English
   audio is kept as a second track **only** when you pass `--keep-original-audio`).
6. Re-run the same command — every stage should print "skipping (done)".
7. Force a re-translate: `uv run vpm2 "<url>" --force translate`.
8. Compare translation models: re-run with a different model and force the translate
   stage, e.g. `uv run vpm2 "<url>" --ollama-model zongwei/gemma3-translator --force translate`.
