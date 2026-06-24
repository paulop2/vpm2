# VPM2 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI that takes a YouTube URL and produces an MP4 with the original English speech replaced by a synced Portuguese (PT-BR) dub.

**Architecture:** A staged pipeline (`download → extract_audio → transcribe → translate → synthesize → assemble`). Each stage reads/writes artifacts in `work/<video-id>/`, is resumable (skip if its output already exists and validates), and can be swapped without touching the orchestrator. Pure logic (timeline sync, artifact validation, prompt building, resume logic) is unit-tested; external-tool/model stages are thin wrappers verified by a manual smoke test.

**Tech Stack:** Python 3.11+, `uv` (env + deps), `yt-dlp`, `ffmpeg` (external binary), `faster-whisper` (ASR), `Ollama` + `requests` (translation LLM), `chatterbox-tts` (TTS, XTTS-v2 as future fallback), `soundfile` + `numpy` (audio assembly), `pytest` (tests). Runs in WSL2 (Ubuntu) with GPU.

## Global Constraints

- Python managed exclusively with **`uv`** (`uv add`, `uv run`); never `pip install` directly.
- Runs in **WSL2 Ubuntu** with NVIDIA GPU (RTX 5060 Ti, Blackwell/sm_120). PyTorch must be a **CUDA 12.8+ (cu128)** build.
- All intermediate artifacts live in `work/<video-id>/`; this directory is **gitignored**.
- v1 assumes a **single speaker** and **one strong ASR** (no ensemble, no diarization).
- TTS primary model is **Chatterbox Multilingual**; the `TTSBackend` interface must allow an XTTS-v2 fallback without pipeline changes.
- Models are loaded on demand and released between heavy stages (ASR → LLM → TTS run **sequentially**).
- Target language is **PT-BR**; source is **English**.
- Sync acceleration cap default **1.25x**; never exceed without config change.

---

## File Structure

```
vpm2/
├─ pyproject.toml                 # uv project, deps, CLI entrypoint
├─ .gitignore                     # already present (work/, *.mp4, *.wav, .venv/)
├─ README.md                      # setup + smoke test
├─ vpm2/
│  ├─ __init__.py
│  ├─ cli.py                      # argparse entrypoint -> run_pipeline
│  ├─ context.py                  # Context dataclass + path helpers
│  ├─ config.py                   # Config dataclass (models, lang, voice, sync params)
│  ├─ pipeline.py                 # STAGES list + run_pipeline (skip/resume/force)
│  ├─ artifacts.py                # JSON read/write + schema validation helpers
│  ├─ timeline.py                 # PURE sync algorithm (plan_timeline)
│  ├─ stages/
│  │  ├─ __init__.py
│  │  ├─ base.py                  # Stage ABC
│  │  ├─ download.py              # 01 yt-dlp
│  │  ├─ extract_audio.py         # 02 ffmpeg
│  │  ├─ transcribe.py            # 03 faster-whisper
│  │  ├─ translate.py             # 04 Ollama LLM
│  │  ├─ synthesize.py            # 05 TTS
│  │  └─ assemble.py              # 06 sync + mux
│  ├─ translate_prompt.py         # PURE prompt builder for translation
│  ├─ voice_sample.py             # PURE-ish: pick clean reference window from VAD spans
│  └─ tts/
│     ├─ __init__.py
│     ├─ base.py                  # TTSBackend ABC + factory
│     └─ chatterbox_backend.py    # Chatterbox impl (preset + cloning modes)
└─ tests/
   ├─ test_pipeline.py
   ├─ test_artifacts.py
   ├─ test_timeline.py
   ├─ test_translate_prompt.py
   └─ test_voice_sample.py
```

## Data Model (artifact schemas)

- `work/<id>/01_video.mp4`, `work/<id>/01_meta.json`
  `{"id": str, "title": str, "url": str, "duration": float}`
- `work/<id>/02_audio.wav` — 16 kHz mono PCM
- `work/<id>/03_transcript.json`
  `{"language": "en", "segments": [{"id": int, "start": float, "end": float, "text": str}, ...]}`
- `work/<id>/04_translation.json`
  `{"segments": [{"id": int, "start": float, "end": float, "text": str, "text_pt": str}, ...]}`
- `work/<id>/05_clips/0000.wav, 0001.wav, ...` and `work/<id>/05_clips.json`
  `{"sample_rate": int, "segments": [{"id": int, "start": float, "end": float, "clip": "0000.wav", "duration": float}, ...]}`
- `work/<id>/06_audio_pt.wav`, `work/<id>/06_final.mp4`
- `work/<id>/ref_voice.wav` — extracted cloning reference (cloning mode only)
- `work/<id>/logs/<stage>.log`

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `vpm2/__init__.py`, `vpm2/stages/__init__.py`, `vpm2/tts/__init__.py`, `tests/__init__.py`
- Create: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable package `vpm2` with console script `vpm2 = "vpm2.cli:main"`; `uv run pytest` works.

- [ ] **Step 1: Initialize uv project**

Run:
```bash
cd /mnt/c/Users/PVS/projetos/vpm2
uv init --package --name vpm2 --python 3.11
```
Expected: creates `pyproject.toml` and `src/`-less package layout. If `uv init` created a `vpm2/` or `src/vpm2/` with a sample module, keep the package dir at top-level `vpm2/` (move if needed) so paths in this plan match.

- [ ] **Step 2: Add runtime dependencies**

Run:
```bash
uv add yt-dlp requests soundfile numpy faster-whisper
uv add --dev pytest
```
Expected: dependencies resolve and `uv.lock` is written. (PyTorch/Chatterbox are added in their own tasks because of the CUDA index.)

- [ ] **Step 3: Define console entrypoint**

Edit `pyproject.toml`, add under `[project.scripts]`:
```toml
[project.scripts]
vpm2 = "vpm2.cli:main"
```

- [ ] **Step 4: Create package marker files**

Create `vpm2/__init__.py`:
```python
__version__ = "0.1.0"
```
Create empty `vpm2/stages/__init__.py`, `vpm2/tts/__init__.py`, `tests/__init__.py` (empty files).

- [ ] **Step 5: Write README skeleton**

Create `README.md`:
```markdown
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
```

- [ ] **Step 6: Verify package imports**

Run: `uv run python -c "import vpm2; print(vpm2.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold uv project for vpm2"
```

---

### Task 2: Config and Context

**Files:**
- Create: `vpm2/config.py`, `vpm2/context.py`

**Interfaces:**
- Produces:
  - `Config` dataclass with fields:
    `asr_model: str = "large-v3"`, `ollama_model: str = "qwen2.5:7b-instruct"`,
    `ollama_url: str = "http://localhost:11434"`, `tts_backend: str = "chatterbox"`,
    `voice_mode: str = "preset"` (`"preset"` | `"cloning"`), `preset_ref_wav: str | None = None`,
    `source_lang: str = "en"`, `target_lang: str = "pt"`,
    `max_speed: float = 1.25`, `allow_push: bool = True`, `keep_original_audio: bool = True`.
  - `Context` dataclass: `url: str`, `work_dir: Path`, `config: Config`, with method
    `path(self, name: str) -> Path` returning `self.work_dir / name`, and
    `log_dir(self) -> Path` returning `self.work_dir / "logs"` (created on access).

- [ ] **Step 1: Write Config**

Create `vpm2/config.py`:
```python
from dataclasses import dataclass


@dataclass
class Config:
    asr_model: str = "large-v3"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    tts_backend: str = "chatterbox"
    voice_mode: str = "preset"          # "preset" | "cloning"
    preset_ref_wav: str | None = None   # path to curated PT reference clip
    source_lang: str = "en"
    target_lang: str = "pt"
    max_speed: float = 1.25
    allow_push: bool = True
    keep_original_audio: bool = True
```

- [ ] **Step 2: Write Context**

Create `vpm2/context.py`:
```python
from dataclasses import dataclass
from pathlib import Path

from vpm2.config import Config


@dataclass
class Context:
    url: str
    work_dir: Path
    config: Config

    def path(self, name: str) -> Path:
        return self.work_dir / name

    def log_dir(self) -> Path:
        d = self.work_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d
```

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from vpm2.context import Context; from vpm2.config import Config; print(Config().max_speed)"`
Expected: prints `1.25`.

- [ ] **Step 4: Commit**

```bash
git add vpm2/config.py vpm2/context.py
git commit -m "feat: add Config and Context"
```

---

### Task 3: Artifact helpers (JSON I/O + validation)

**Files:**
- Create: `vpm2/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Produces:
  - `write_json(path: Path, data: dict) -> None`
  - `read_json(path: Path) -> dict`
  - `valid_transcript(path: Path) -> bool` — file exists, parses, has non-empty `segments`, each with `id/start/end/text`.
  - `valid_translation(path: Path) -> bool` — like transcript plus non-empty `text_pt` on every segment.
  - `valid_clips(path: Path, clips_dir: Path) -> bool` — parses, every segment's `clip` file exists in `clips_dir`, count > 0.

- [ ] **Step 1: Write failing tests**

Create `tests/test_artifacts.py`:
```python
import json
from pathlib import Path

from vpm2.artifacts import (
    read_json, write_json,
    valid_transcript, valid_translation, valid_clips,
)


def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1})
    assert read_json(p) == {"a": 1}


def test_valid_transcript_true(tmp_path):
    p = tmp_path / "t.json"
    write_json(p, {"language": "en", "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}]})
    assert valid_transcript(p) is True


def test_valid_transcript_false_when_empty(tmp_path):
    p = tmp_path / "t.json"
    write_json(p, {"language": "en", "segments": []})
    assert valid_transcript(p) is False


def test_valid_transcript_false_when_missing_file(tmp_path):
    assert valid_transcript(tmp_path / "nope.json") is False


def test_valid_translation_requires_text_pt(tmp_path):
    p = tmp_path / "tr.json"
    write_json(p, {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "text_pt": ""}]})
    assert valid_translation(p) is False
    write_json(p, {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "text_pt": "oi"}]})
    assert valid_translation(p) is True


def test_valid_clips_checks_files_exist(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "0000.wav").write_bytes(b"RIFF")
    p = tmp_path / "c.json"
    write_json(p, {"sample_rate": 24000, "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "clip": "0000.wav", "duration": 0.9}]})
    assert valid_clips(p, clips) is True
    # missing file -> invalid
    write_json(p, {"sample_rate": 24000, "segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "clip": "9999.wav", "duration": 0.9}]})
    assert valid_clips(p, clips) is False
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vpm2.artifacts'`.

- [ ] **Step 3: Implement artifacts.py**

Create `vpm2/artifacts.py`:
```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vpm2/artifacts.py tests/test_artifacts.py
git commit -m "feat: add artifact JSON I/O and validation helpers"
```

---

### Task 4: Timeline sync algorithm (PURE — core logic)

**Files:**
- Create: `vpm2/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Produces:
  - `@dataclass PlacedClip{ id: int, start: float, speed: float, src_duration: float, out_duration: float }`
  - `plan_timeline(segments: list[dict], video_duration: float, max_speed: float = 1.25, allow_push: bool = True) -> list[PlacedClip]`
    where each `segments` item has `id: int`, `start: float`, `end: float`, `duration: float` (the TTS clip's real length).
    Rules: clips are placed in `id` order; a clip starts no earlier than `start` and no earlier than the previous clip's end (push). The space available is until the next segment's original `start` (or `video_duration` for the last). If the clip overruns, accelerate up to `max_speed`; if still overrunning and `allow_push`, let it overrun and push the cursor.

- [ ] **Step 1: Write failing tests**

Create `tests/test_timeline.py`:
```python
import pytest

from vpm2.timeline import plan_timeline, PlacedClip


def seg(i, start, end, duration):
    return {"id": i, "start": start, "end": end, "duration": duration}


def test_clip_fits_within_gap_plays_at_normal_speed():
    # segment 0..4 (gap to next at 5.0), clip is 3.0s -> fits
    segs = [seg(0, 0.0, 4.0, 3.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=10.0)
    assert placed[0] == PlacedClip(id=0, start=0.0, speed=1.0,
                                   src_duration=3.0, out_duration=3.0)


def test_clip_uses_pause_before_next_segment():
    # clip 4.5s, next segment starts at 5.0 -> 5.0s available, still fits
    segs = [seg(0, 0.0, 4.0, 4.5), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=10.0)
    assert placed[0].speed == 1.0
    assert placed[0].out_duration == 4.5


def test_overrun_accelerates_within_cap():
    # clip 6.0s, available 5.0s -> speed 1.2 (<=1.25), out 5.0s
    segs = [seg(0, 0.0, 4.0, 6.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=10.0, max_speed=1.25)
    assert placed[0].speed == pytest.approx(1.2)
    assert placed[0].out_duration == pytest.approx(5.0)


def test_overrun_beyond_cap_clamps_and_pushes_next():
    # clip 10.0s, available 5.0s -> needs 2.0x but capped 1.25 -> out 8.0s
    segs = [seg(0, 0.0, 4.0, 10.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=20.0,
                           max_speed=1.25, allow_push=True)
    assert placed[0].speed == pytest.approx(1.25)
    assert placed[0].out_duration == pytest.approx(8.0)
    # next clip pushed to start at 0 + 8.0 = 8.0 (after segment0 end on timeline)
    assert placed[1].start == pytest.approx(8.0)


def test_last_segment_bounded_by_video_duration():
    segs = [seg(0, 8.0, 9.0, 4.0)]
    placed = plan_timeline(segs, video_duration=10.0, max_speed=1.25)
    # available = 10.0 - 8.0 = 2.0, clip 4.0 -> needs 2.0x capped 1.25 -> out 3.2
    assert placed[0].start == pytest.approx(8.0)
    assert placed[0].speed == pytest.approx(1.25)
    assert placed[0].out_duration == pytest.approx(3.2)


def test_no_push_keeps_next_at_original_start():
    segs = [seg(0, 0.0, 4.0, 10.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=20.0,
                           max_speed=1.25, allow_push=False)
    assert placed[1].start == pytest.approx(5.0)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_timeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vpm2.timeline'`.

- [ ] **Step 3: Implement timeline.py**

Create `vpm2/timeline.py`:
```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_timeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vpm2/timeline.py tests/test_timeline.py
git commit -m "feat: add pure timeline sync algorithm"
```

---

### Task 5: Stage base + pipeline orchestration (resume/force)

**Files:**
- Create: `vpm2/stages/base.py`, `vpm2/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `class Stage(ABC)` with `name: str` (class attr), abstract `output_path(ctx) -> Path`, `is_done(ctx) -> bool`, `run(ctx) -> None`.
  - `run_pipeline(ctx: Context, stages: list[Stage] | None = None, force_from: str | None = None) -> None`
    Iterates stages in order; for each: if `force_from` is set and we have reached that stage (or passed it), always `run`; otherwise `run` only if `not is_done(ctx)`. Records which stages ran for testability via return value `list[str]` of executed stage names.
  - Module-level `STAGES: list[Stage]` (populated in later tasks; starts empty import-safe).

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:
```python
from pathlib import Path

from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.pipeline import run_pipeline


class FakeStage(Stage):
    def __init__(self, name, done):
        self.name = name
        self._done = done
        self.ran = False

    def output_path(self, ctx):
        return ctx.path(f"{self.name}.txt")

    def is_done(self, ctx):
        return self._done

    def run(self, ctx):
        self.ran = True


def make_ctx(tmp_path):
    return Context(url="x", work_dir=tmp_path, config=Config())


def test_runs_only_not_done_stages(tmp_path):
    a = FakeStage("a", done=True)
    b = FakeStage("b", done=False)
    executed = run_pipeline(make_ctx(tmp_path), stages=[a, b])
    assert executed == ["b"]
    assert a.ran is False and b.ran is True


def test_force_from_reruns_that_stage_onward(tmp_path):
    a = FakeStage("a", done=True)
    b = FakeStage("b", done=True)
    c = FakeStage("c", done=True)
    executed = run_pipeline(make_ctx(tmp_path), stages=[a, b, c], force_from="b")
    assert executed == ["b", "c"]
    assert a.ran is False and b.ran is True and c.ran is True
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError: vpm2.stages.base`).

- [ ] **Step 3: Implement Stage base**

Create `vpm2/stages/base.py`:
```python
from abc import ABC, abstractmethod
from pathlib import Path

from vpm2.context import Context


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def output_path(self, ctx: Context) -> Path: ...

    @abstractmethod
    def is_done(self, ctx: Context) -> bool: ...

    @abstractmethod
    def run(self, ctx: Context) -> None: ...
```

- [ ] **Step 4: Implement pipeline**

Create `vpm2/pipeline.py`:
```python
from vpm2.context import Context
from vpm2.stages.base import Stage

# Concrete stages are appended in their own tasks to keep imports light here.
STAGES: list[Stage] = []


def run_pipeline(ctx: Context, stages: list[Stage] | None = None,
                 force_from: str | None = None) -> list[str]:
    stages = STAGES if stages is None else stages
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    forcing = False
    executed: list[str] = []
    for stage in stages:
        if force_from is not None and stage.name == force_from:
            forcing = True
        if forcing or not stage.is_done(ctx):
            print(f"[vpm2] running stage: {stage.name}")
            stage.run(ctx)
            executed.append(stage.name)
        else:
            print(f"[vpm2] skipping (done): {stage.name}")
    return executed
```

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add vpm2/stages/base.py vpm2/pipeline.py tests/test_pipeline.py
git commit -m "feat: add Stage base and pipeline orchestration with resume/force"
```

---

### Task 6: Translation prompt builder (PURE)

**Files:**
- Create: `vpm2/translate_prompt.py`
- Test: `tests/test_translate_prompt.py`

**Interfaces:**
- Produces:
  - `build_translation_prompt(segment_text: str, prev_text: str | None, next_text: str | None) -> str`
    Returns an instruction prompt that asks for a **natural, spoken PT-BR** translation of `segment_text` only, using neighbors as context, and instructs the model to return ONLY the translated line (no quotes, no notes).

- [ ] **Step 1: Write failing tests**

Create `tests/test_translate_prompt.py`:
```python
from vpm2.translate_prompt import build_translation_prompt


def test_prompt_contains_segment_and_context():
    p = build_translation_prompt("Let's paint a tree.",
                                 prev_text="Hello there.",
                                 next_text="A happy little tree.")
    assert "Let's paint a tree." in p
    assert "Hello there." in p
    assert "A happy little tree." in p


def test_prompt_handles_missing_neighbors():
    p = build_translation_prompt("Hi.", prev_text=None, next_text=None)
    assert "Hi." in p
    # must not crash and must still ask for PT-BR
    assert "português" in p.lower() or "pt-br" in p.lower()


def test_prompt_requests_only_translation():
    p = build_translation_prompt("Hi.", None, None)
    assert "apenas" in p.lower() or "only" in p.lower()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_translate_prompt.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement translate_prompt.py**

Create `vpm2/translate_prompt.py`:
```python
def build_translation_prompt(segment_text, prev_text=None, next_text=None):
    context_lines = []
    if prev_text:
        context_lines.append(f"Frase anterior (contexto): {prev_text}")
    if next_text:
        context_lines.append(f"Próxima frase (contexto): {next_text}")
    context = "\n".join(context_lines)
    context_block = f"{context}\n\n" if context else ""

    return (
        "Você é um tradutor de legendas para dublagem. Traduza para "
        "PORTUGUÊS (PT-BR) de forma natural e falada, como uma narração — "
        "não literal, mas fiel ao sentido. Mantenha o comprimento parecido "
        "com o original quando possível.\n\n"
        f"{context_block}"
        "Traduza APENAS a frase abaixo. Responda somente com a tradução, "
        "sem aspas, sem comentários, sem o texto original.\n\n"
        f"Frase: {segment_text}"
    )
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_translate_prompt.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vpm2/translate_prompt.py tests/test_translate_prompt.py
git commit -m "feat: add translation prompt builder"
```

---

### Task 7: Voice sample picker (PURE)

**Files:**
- Create: `vpm2/voice_sample.py`
- Test: `tests/test_voice_sample.py`

**Interfaces:**
- Produces:
  - `pick_reference_window(speech_spans: list[tuple[float, float]], target: float = 10.0, min_len: float = 6.0) -> tuple[float, float] | None`
    Given speech spans `(start, end)` (seconds) detected by VAD, choose a window of up to `target` seconds inside the **longest** span (so it is contiguous speech, no music/silence). Returns `(start, end)` or `None` if no span reaches `min_len`. The window starts a little inside the span (skip 0.25s onset) and never exceeds the span end.

- [ ] **Step 1: Write failing tests**

Create `tests/test_voice_sample.py`:
```python
from vpm2.voice_sample import pick_reference_window


def test_picks_inside_longest_span():
    spans = [(0.0, 2.0), (10.0, 30.0), (31.0, 33.0)]
    win = pick_reference_window(spans, target=10.0, min_len=6.0)
    assert win is not None
    start, end = win
    assert 10.0 <= start < end <= 30.0
    assert end - start <= 10.0
    assert end - start >= 6.0


def test_returns_none_when_no_span_long_enough():
    spans = [(0.0, 2.0), (5.0, 9.0)]
    assert pick_reference_window(spans, target=10.0, min_len=6.0) is None


def test_window_clamped_to_span_end():
    spans = [(0.0, 7.0)]  # 7s span, target 10 -> clamp to span end
    win = pick_reference_window(spans, target=10.0, min_len=6.0)
    assert win is not None
    start, end = win
    assert end <= 7.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_voice_sample.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement voice_sample.py**

Create `vpm2/voice_sample.py`:
```python
def pick_reference_window(speech_spans, target=10.0, min_len=6.0):
    if not speech_spans:
        return None
    longest = max(speech_spans, key=lambda s: s[1] - s[0])
    span_start, span_end = longest
    if (span_end - span_start) < min_len:
        return None
    start = min(span_start + 0.25, span_end - min_len)
    start = max(start, span_start)
    end = min(start + target, span_end)
    return (start, end)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_voice_sample.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vpm2/voice_sample.py tests/test_voice_sample.py
git commit -m "feat: add voice reference window picker"
```

---

### Task 8: Download stage (yt-dlp)

**Files:**
- Create: `vpm2/stages/download.py`
- Modify: `vpm2/pipeline.py` (register stage)

**Interfaces:**
- Consumes: `Context.url`.
- Produces: `01_video.mp4`, `01_meta.json` (`{id, title, url, duration}`). `DownloadStage.name == "download"`.

This stage is a thin wrapper over `yt-dlp`'s Python API. No unit test (verified by smoke test); keep logic minimal.

- [ ] **Step 1: Implement download stage**

Create `vpm2/stages/download.py`:
```python
from pathlib import Path

import yt_dlp

from vpm2.artifacts import write_json
from vpm2.context import Context
from vpm2.stages.base import Stage


class DownloadStage(Stage):
    name = "download"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("01_video.mp4")

    def is_done(self, ctx: Context) -> bool:
        return self.output_path(ctx).exists() and ctx.path("01_meta.json").exists()

    def run(self, ctx: Context) -> None:
        out = self.output_path(ctx)
        opts = {
            "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(out),
            "quiet": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(ctx.url, download=True)
        write_json(ctx.path("01_meta.json"), {
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "url": ctx.url,
            "duration": float(info.get("duration") or 0.0),
        })
```

- [ ] **Step 2: Register the stage**

Edit `vpm2/pipeline.py`, replace `STAGES: list[Stage] = []` with:
```python
from vpm2.stages.download import DownloadStage

STAGES: list[Stage] = [
    DownloadStage(),
]
```

- [ ] **Step 3: Verify import + pipeline tests still pass**

Run: `uv run pytest tests/test_pipeline.py -v && uv run python -c "from vpm2.pipeline import STAGES; print([s.name for s in STAGES])"`
Expected: tests PASS; prints `['download']`.

- [ ] **Step 4: Commit**

```bash
git add vpm2/stages/download.py vpm2/pipeline.py
git commit -m "feat: add download stage (yt-dlp)"
```

---

### Task 9: Extract-audio stage (ffmpeg)

**Files:**
- Create: `vpm2/stages/extract_audio.py`
- Modify: `vpm2/pipeline.py`

**Interfaces:**
- Consumes: `01_video.mp4`.
- Produces: `02_audio.wav` (16 kHz mono). `ExtractAudioStage.name == "extract_audio"`.

- [ ] **Step 1: Implement extract-audio stage**

Create `vpm2/stages/extract_audio.py`:
```python
import subprocess
from pathlib import Path

from vpm2.context import Context
from vpm2.stages.base import Stage


class ExtractAudioStage(Stage):
    name = "extract_audio"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("02_audio.wav")

    def is_done(self, ctx: Context) -> bool:
        return self.output_path(ctx).exists()

    def run(self, ctx: Context) -> None:
        video = ctx.path("01_video.mp4")
        out = self.output_path(ctx)
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out),
        ]
        log = ctx.log_dir() / "extract_audio.log"
        with open(log, "w") as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
```

- [ ] **Step 2: Register the stage**

Edit `vpm2/pipeline.py`:
```python
from vpm2.stages.download import DownloadStage
from vpm2.stages.extract_audio import ExtractAudioStage

STAGES: list[Stage] = [
    DownloadStage(),
    ExtractAudioStage(),
]
```

- [ ] **Step 3: Verify**

Run: `uv run python -c "from vpm2.pipeline import STAGES; print([s.name for s in STAGES])"`
Expected: prints `['download', 'extract_audio']`.

- [ ] **Step 4: Commit**

```bash
git add vpm2/stages/extract_audio.py vpm2/pipeline.py
git commit -m "feat: add extract-audio stage (ffmpeg)"
```

---

### Task 10: Transcribe stage (faster-whisper)

**Files:**
- Create: `vpm2/stages/transcribe.py`
- Modify: `vpm2/pipeline.py`

**Interfaces:**
- Consumes: `02_audio.wav`, `Config.asr_model`, `Config.source_lang`.
- Produces: `03_transcript.json` (schema in Data Model). `TranscribeStage.name == "transcribe"`. Uses `valid_transcript` for `is_done`.

- [ ] **Step 1: Verify faster-whisper API**

Run: `uv run python -c "from faster_whisper import WhisperModel; print('ok')"`
Expected: prints `ok`. If import fails, ensure Task 1 added `faster-whisper`. The API used below: `WhisperModel(model_size, device="cuda", compute_type="float16")` and `model.transcribe(path, language="en")` returns `(segments, info)` where each segment has `.start`, `.end`, `.text`.

- [ ] **Step 2: Implement transcribe stage**

Create `vpm2/stages/transcribe.py`:
```python
from pathlib import Path

from faster_whisper import WhisperModel

from vpm2.artifacts import valid_transcript, write_json
from vpm2.context import Context
from vpm2.stages.base import Stage


class TranscribeStage(Stage):
    name = "transcribe"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("03_transcript.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_transcript(self.output_path(ctx))

    def run(self, ctx: Context) -> None:
        model = WhisperModel(
            ctx.config.asr_model, device="cuda", compute_type="float16",
        )
        segments, _info = model.transcribe(
            str(ctx.path("02_audio.wav")),
            language=ctx.config.source_lang,
            vad_filter=True,
        )
        out_segments = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            if not text:
                continue
            out_segments.append({
                "id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
        write_json(self.output_path(ctx), {
            "language": ctx.config.source_lang,
            "segments": out_segments,
        })
        del model
```

- [ ] **Step 3: Register the stage**

Edit `vpm2/pipeline.py` to add `from vpm2.stages.transcribe import TranscribeStage` and append `TranscribeStage()` to `STAGES`.

- [ ] **Step 4: Verify**

Run: `uv run python -c "from vpm2.pipeline import STAGES; print([s.name for s in STAGES])"`
Expected: prints `['download', 'extract_audio', 'transcribe']`.

- [ ] **Step 5: Commit**

```bash
git add vpm2/stages/transcribe.py vpm2/pipeline.py
git commit -m "feat: add transcribe stage (faster-whisper)"
```

---

### Task 11: Translate stage (Ollama)

**Files:**
- Create: `vpm2/stages/translate.py`
- Modify: `vpm2/pipeline.py`

**Interfaces:**
- Consumes: `03_transcript.json`, `Config.ollama_model`, `Config.ollama_url`, `build_translation_prompt`.
- Produces: `04_translation.json`. `TranslateStage.name == "translate"`. Uses `valid_translation` for `is_done`.

- [ ] **Step 1: Implement translate stage**

Create `vpm2/stages/translate.py`:
```python
from pathlib import Path

import requests

from vpm2.artifacts import read_json, valid_translation, write_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.translate_prompt import build_translation_prompt


class TranslateStage(Stage):
    name = "translate"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("04_translation.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_translation(self.output_path(ctx))

    def _translate_one(self, ctx, text, prev, nxt):
        prompt = build_translation_prompt(text, prev, nxt)
        resp = requests.post(
            f"{ctx.config.ollama_url}/api/generate",
            json={"model": ctx.config.ollama_model,
                  "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def run(self, ctx: Context) -> None:
        data = read_json(ctx.path("03_transcript.json"))
        segs = data["segments"]
        out = []
        for i, s in enumerate(segs):
            prev = segs[i - 1]["text"] if i > 0 else None
            nxt = segs[i + 1]["text"] if i + 1 < len(segs) else None
            text_pt = self._translate_one(ctx, s["text"], prev, nxt)
            out.append({**s, "text_pt": text_pt})
        write_json(self.output_path(ctx), {"segments": out})
```

- [ ] **Step 2: Register the stage**

Edit `vpm2/pipeline.py` to import and append `TranslateStage()`.

- [ ] **Step 3: Verify**

Run: `uv run python -c "from vpm2.pipeline import STAGES; print([s.name for s in STAGES])"`
Expected: prints the four stage names ending with `translate`.

- [ ] **Step 4: Commit**

```bash
git add vpm2/stages/translate.py vpm2/pipeline.py
git commit -m "feat: add translate stage (Ollama)"
```

---

### Task 12: TTS backend interface + Chatterbox

**Files:**
- Create: `vpm2/tts/base.py`, `vpm2/tts/chatterbox_backend.py`

**Interfaces:**
- Produces:
  - `class TTSBackend(ABC)` with `sample_rate: int` and abstract
    `synth(self, text: str, ref_wav: Path | None) -> "numpy.ndarray"` (float32 mono at `sample_rate`).
  - `get_backend(config: Config) -> TTSBackend` factory: returns `ChatterboxBackend` for `config.tts_backend == "chatterbox"`, else raises `ValueError` (XTTS fallback is a future task).
  - `ChatterboxBackend`: loads model lazily on first `synth`; in cloning mode `ref_wav` is the extracted reference, in preset mode `ref_wav` is `config.preset_ref_wav`.

This wraps the Chatterbox model; verify the exact API before coding.

- [ ] **Step 1: Add PyTorch (cu128) and Chatterbox**

Run:
```bash
uv add torch torchaudio --index https://download.pytorch.org/whl/cu128
uv add chatterbox-tts
```
Expected: resolves. Then verify GPU:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
Expected: `True 12.8` (or similar). If `False`, fix the WSL CUDA/driver setup before continuing.

- [ ] **Step 2: Verify Chatterbox API**

Run:
```bash
uv run python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print('ok')"
```
Expected: `ok`. Confirm the generation call signature in the installed version (README/`help`): the code below assumes
`model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")` and
`wav = model.generate(text, language_id="pt", audio_prompt_path=str(ref))` returning a torch tensor at `model.sr`. If the installed API differs (method name, kwargs), adjust `synth` accordingly — keep the `TTSBackend.synth` signature unchanged.

- [ ] **Step 3: Implement TTSBackend base + factory**

Create `vpm2/tts/base.py`:
```python
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from vpm2.config import Config


class TTSBackend(ABC):
    sample_rate: int = 24000

    @abstractmethod
    def synth(self, text: str, ref_wav: Path | None) -> np.ndarray:
        """Return float32 mono audio at self.sample_rate."""


def get_backend(config: Config) -> TTSBackend:
    if config.tts_backend == "chatterbox":
        from vpm2.tts.chatterbox_backend import ChatterboxBackend
        return ChatterboxBackend(config)
    raise ValueError(f"unknown tts backend: {config.tts_backend}")
```

- [ ] **Step 4: Implement Chatterbox backend**

Create `vpm2/tts/chatterbox_backend.py`:
```python
from pathlib import Path

import numpy as np

from vpm2.config import Config
from vpm2.tts.base import TTSBackend


class ChatterboxBackend(TTSBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            self.sample_rate = int(self._model.sr)

    def synth(self, text: str, ref_wav: Path | None) -> np.ndarray:
        self._ensure_model()
        kwargs = {"language_id": self._config.target_lang}
        if ref_wav is not None:
            kwargs["audio_prompt_path"] = str(ref_wav)
        wav = self._model.generate(text, **kwargs)
        arr = wav.squeeze().detach().cpu().numpy().astype("float32")
        return arr
```

- [ ] **Step 5: Verify imports (no model download)**

Run: `uv run python -c "from vpm2.tts.base import get_backend; from vpm2.config import Config; print(type(get_backend(Config())).__name__)"`
Expected: prints `ChatterboxBackend`.

- [ ] **Step 6: Commit**

```bash
git add vpm2/tts/base.py vpm2/tts/chatterbox_backend.py pyproject.toml uv.lock
git commit -m "feat: add TTS backend interface and Chatterbox implementation"
```

---

### Task 13: Synthesize stage (TTS + cloning reference extraction)

**Files:**
- Create: `vpm2/stages/synthesize.py`
- Modify: `vpm2/pipeline.py`

**Interfaces:**
- Consumes: `04_translation.json`, `02_audio.wav`, `TTSBackend`, `pick_reference_window`, `Config.voice_mode`/`preset_ref_wav`.
- Produces: `05_clips/NNNN.wav` (one per segment) + `05_clips.json`. In cloning mode also writes `ref_voice.wav`. `SynthesizeStage.name == "synthesize"`. Uses `valid_clips` for `is_done`.

- [ ] **Step 1: Implement reference extraction helper**

In `vpm2/stages/synthesize.py`, the cloning reference is extracted by running faster-whisper's VAD over `02_audio.wav` to get speech spans, then `pick_reference_window`, then slicing the wav with `soundfile`. Create `vpm2/stages/synthesize.py`:
```python
from pathlib import Path

import numpy as np
import soundfile as sf

from vpm2.artifacts import read_json, valid_clips, write_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.tts.base import get_backend
from vpm2.voice_sample import pick_reference_window


def _extract_reference(ctx: Context) -> Path:
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    audio_path = ctx.path("02_audio.wav")
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    # faster-whisper VAD expects 16kHz float32
    ts = get_speech_timestamps(data.astype("float32"), VadOptions())
    spans = [(t["start"] / sr, t["end"] / sr) for t in ts]
    win = pick_reference_window(spans)
    ref_path = ctx.path("ref_voice.wav")
    if win is None:
        # fallback: first 10s
        start, end = 0.0, min(10.0, len(data) / sr)
    else:
        start, end = win
    sf.write(str(ref_path), data[int(start * sr):int(end * sr)], sr)
    return ref_path


class SynthesizeStage(Stage):
    name = "synthesize"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("05_clips.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_clips(self.output_path(ctx), ctx.path("05_clips"))

    def run(self, ctx: Context) -> None:
        clips_dir = ctx.path("05_clips")
        clips_dir.mkdir(parents=True, exist_ok=True)

        if ctx.config.voice_mode == "cloning":
            ref = _extract_reference(ctx)
        elif ctx.config.preset_ref_wav:
            ref = Path(ctx.config.preset_ref_wav)
        else:
            ref = None

        backend = get_backend(ctx.config)
        segs = read_json(ctx.path("04_translation.json"))["segments"]
        out = []
        for s in segs:
            audio = backend.synth(s["text_pt"], ref)
            name = f"{s['id']:04d}.wav"
            sf.write(str(clips_dir / name), audio, backend.sample_rate)
            out.append({
                "id": s["id"], "start": s["start"], "end": s["end"],
                "clip": name, "duration": len(audio) / backend.sample_rate,
            })
        write_json(self.output_path(ctx), {
            "sample_rate": backend.sample_rate, "segments": out,
        })
```

- [ ] **Step 2: Verify VAD import path**

Run: `uv run python -c "from faster_whisper.vad import get_speech_timestamps, VadOptions; print('ok')"`
Expected: `ok`. If the import path differs in the installed version, adjust `_extract_reference` to the correct VAD helper (the rest of the stage is unaffected).

- [ ] **Step 3: Register the stage**

Edit `vpm2/pipeline.py` to import and append `SynthesizeStage()`.

- [ ] **Step 4: Verify**

Run: `uv run python -c "from vpm2.pipeline import STAGES; print([s.name for s in STAGES])"`
Expected: five stage names ending with `synthesize`.

- [ ] **Step 5: Commit**

```bash
git add vpm2/stages/synthesize.py vpm2/pipeline.py
git commit -m "feat: add synthesize stage (TTS + cloning reference extraction)"
```

---

### Task 14: Assemble stage (sync + mux)

**Files:**
- Create: `vpm2/stages/assemble.py`
- Modify: `vpm2/pipeline.py`

**Interfaces:**
- Consumes: `05_clips.json`, `05_clips/*.wav`, `01_video.mp4`, `01_meta.json`, `plan_timeline`, `Config` sync params.
- Produces: `06_audio_pt.wav`, `06_final.mp4`. `AssembleStage.name == "assemble"`.

Approach: use `plan_timeline` to compute placement + speed; apply `ffmpeg atempo` per clip needing speedup; place each clip into a single numpy buffer at its sample offset; write `06_audio_pt.wav`; mux with ffmpeg (replace audio; keep original as second muted track if `keep_original_audio`).

- [ ] **Step 1: Implement assemble stage**

Create `vpm2/stages/assemble.py`:
```python
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from vpm2.artifacts import read_json
from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.timeline import plan_timeline


def _atempo(in_path: Path, out_path: Path, speed: float) -> None:
    # ffmpeg atempo supports 0.5..2.0 per filter; our cap is <=1.25 so one pass.
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-filter:a", f"atempo={speed:.4f}", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


class AssembleStage(Stage):
    name = "assemble"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("06_final.mp4")

    def is_done(self, ctx: Context) -> bool:
        return self.output_path(ctx).exists()

    def run(self, ctx: Context) -> None:
        clips_data = read_json(ctx.path("05_clips.json"))
        sr = int(clips_data["sample_rate"])
        clips_dir = ctx.path("05_clips")
        meta = read_json(ctx.path("01_meta.json"))
        video_duration = float(meta["duration"])

        segs = [{"id": s["id"], "start": s["start"], "end": s["end"],
                 "duration": s["duration"]} for s in clips_data["segments"]]
        placed = plan_timeline(
            segs, video_duration,
            max_speed=ctx.config.max_speed, allow_push=ctx.config.allow_push,
        )
        by_id = {s["id"]: s for s in clips_data["segments"]}

        total_samples = int((video_duration + 5.0) * sr)
        buffer = np.zeros(total_samples, dtype="float32")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for pc in placed:
                src = clips_dir / by_id[pc.id]["clip"]
                if pc.speed > 1.001:
                    sped = tmp / f"{pc.id:04d}_s.wav"
                    _atempo(src, sped, pc.speed)
                    audio, csr = sf.read(str(sped))
                else:
                    audio, csr = sf.read(str(src))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if csr != sr:
                    # resample defensively via ffmpeg-less linear interp
                    idx = np.linspace(0, len(audio) - 1,
                                      int(len(audio) * sr / csr))
                    audio = np.interp(idx, np.arange(len(audio)), audio)
                start_sample = int(pc.start * sr)
                end_sample = start_sample + len(audio)
                if end_sample > len(buffer):
                    buffer = np.concatenate(
                        [buffer, np.zeros(end_sample - len(buffer), "float32")])
                buffer[start_sample:end_sample] += audio.astype("float32")

        pt_wav = ctx.path("06_audio_pt.wav")
        sf.write(str(pt_wav), buffer, sr)

        video = ctx.path("01_video.mp4")
        out = self.output_path(ctx)
        if ctx.config.keep_original_audio:
            cmd = [
                "ffmpeg", "-y", "-i", str(video), "-i", str(pt_wav),
                "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
                "-c:v", "copy", "-c:a", "aac",
                "-disposition:a:0", "default", "-disposition:a:1", "none",
                "-shortest", str(out),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(video), "-i", str(pt_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
            ]
        log = ctx.log_dir() / "assemble.log"
        with open(log, "w") as lf:
            subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
```

- [ ] **Step 2: Register the stage**

Edit `vpm2/pipeline.py` to import and append `AssembleStage()`. The final `STAGES` list:
```python
STAGES: list[Stage] = [
    DownloadStage(),
    ExtractAudioStage(),
    TranscribeStage(),
    TranslateStage(),
    SynthesizeStage(),
    AssembleStage(),
]
```

- [ ] **Step 3: Verify all stages registered**

Run: `uv run python -c "from vpm2.pipeline import STAGES; print([s.name for s in STAGES])"`
Expected: `['download', 'extract_audio', 'transcribe', 'translate', 'synthesize', 'assemble']`.

- [ ] **Step 4: Commit**

```bash
git add vpm2/stages/assemble.py vpm2/pipeline.py
git commit -m "feat: add assemble stage (timeline sync + mux)"
```

---

### Task 15: CLI wiring

**Files:**
- Create: `vpm2/cli.py`

**Interfaces:**
- Consumes: `run_pipeline`, `Config`, `Context`, `01_meta.json` id (via yt-dlp pre-extract for the work dir name).
- Produces: `main()` console entrypoint. Flags: `url` (positional), `--voice-mode {preset,cloning}`, `--preset-ref`, `--force <stage>`, `--ollama-model`, `--asr-model`, `--work-root` (default `work`).

The work-dir name needs the video id before download. Resolve it with a metadata-only yt-dlp call; fall back to a slug of the URL.

- [ ] **Step 1: Implement CLI**

Create `vpm2/cli.py`:
```python
import argparse
import re
import sys
from pathlib import Path

import yt_dlp

from vpm2.config import Config
from vpm2.context import Context
from vpm2.pipeline import run_pipeline


def _resolve_id(url: str) -> str:
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        vid = info.get("id")
        if vid:
            return vid
    except Exception:
        pass
    return re.sub(r"[^A-Za-z0-9_-]", "_", url)[-40:]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="vpm2")
    ap.add_argument("url")
    ap.add_argument("--voice-mode", choices=["preset", "cloning"], default="preset")
    ap.add_argument("--preset-ref", default=None)
    ap.add_argument("--force", default=None,
                    help="rerun from this stage name onward")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--work-root", default="work")
    args = ap.parse_args(argv)

    config = Config(voice_mode=args.voice_mode, preset_ref_wav=args.preset_ref)
    if args.ollama_model:
        config.ollama_model = args.ollama_model
    if args.asr_model:
        config.asr_model = args.asr_model

    video_id = _resolve_id(args.url)
    work_dir = Path(args.work_root) / video_id
    ctx = Context(url=args.url, work_dir=work_dir, config=config)

    run_pipeline(ctx, force_from=args.force)
    final = ctx.path("06_final.mp4")
    print(f"[vpm2] done -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify CLI help works**

Run: `uv run vpm2 --help`
Expected: prints usage with the flags above (no network call).

- [ ] **Step 3: Commit**

```bash
git add vpm2/cli.py
git commit -m "feat: add CLI entrypoint"
```

---

### Task 16: Full test pass + README smoke test docs

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documented end-to-end smoke test; green unit suite.

- [ ] **Step 1: Run the whole unit suite**

Run: `uv run pytest -v`
Expected: all tests in `tests/` PASS (timeline, artifacts, pipeline, translate_prompt, voice_sample). No GPU/network needed.

- [ ] **Step 2: Document the smoke test**

Append to `README.md`:
```markdown
## End-to-end smoke test (requires GPU + network)

1. Start Ollama and pull the model: `ollama pull qwen2.5:7b-instruct`
2. Pick a short (~30–60s) English clip URL.
3. Preset voice (provide a clean PT reference wav):
   `uv run vpm2 "<url>" --voice-mode preset --preset-ref ref_pt.wav`
4. Cloning voice (reference auto-extracted from the video):
   `uv run vpm2 "<url>" --voice-mode cloning`
5. Inspect artifacts in `work/<id>/`: open `03_transcript.json`,
   `04_translation.json`, listen to `05_clips/*.wav`, then play `06_final.mp4`.
6. Re-run the same command — every stage should print "skipping (done)".
7. Force a re-translate: `uv run vpm2 "<url>" --force translate`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add end-to-end smoke test instructions"
```

---

## Notes for the implementer

- **Stage order is fixed** in `STAGES`; `--force <stage>` reruns from that name onward (matches the resume design).
- **VRAM:** stages run sequentially and each heavy model is created inside `run()` and dropped when it returns, so ASR/LLM/TTS never coexist. Do not hoist model creation to module scope.
- **External APIs may drift:** Tasks 10, 12, 13 contain explicit verify steps for `faster-whisper`, `chatterbox`, and the VAD import. If the installed signature differs, adjust the call but keep the documented `TTSBackend.synth` / stage interfaces unchanged so neighboring tasks still compose.
- **Future phases (not in this plan):** UI, ASR ensemble + judge, diarization, aggressive sync, XTTS-v2 fallback backend (add a new `vpm2/tts/xtts_backend.py` behind `get_backend`).
