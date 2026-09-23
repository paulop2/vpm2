# VPM2 — Porte de Parakeet ASR + perfis de voz + cache (cópia WSL)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Portar para a cópia canônica do projeto (WSL) o ASR plugável com Parakeet como padrão, perfis de voz persistidos e cache de TTS entre vídeos, preservando `reporter`, `free_cuda`, resume por clipe e escrita atômica já existentes.

**Architecture:** Uma interface `ASRBackend` espelha o `TTSBackend` existente; backends devolvem segmentos crus e uma função pura os normaliza. Um módulo puro de perfis de voz persiste referências e gera chaves de cache; o `SynthesizeStage` existente ganha o modo `profile`, a promoção a perfil e um cache de clipes compartilhado entre vídeos.

**Tech Stack:** Python 3.11+, `uv`, `faster-whisper` (fallback), `nemo_toolkit[asr]` (Parakeet), `chatterbox-tts`, `soundfile`+`numpy`, `rich`, `pytest`.

## Global Constraints

- Repositório canônico: `/home/pvs/projetos/vpm2` (WSL). Branch de trabalho: **`feat/parakeet-asr-voice-profiles`** (já criada a partir de `e915cc7`).
- Os testes rodam **dentro do WSL** (o `.venv` é nativo do WSL). De qualquer shell (inclusive Windows), use:
  `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest <alvo> -q'`
- Shape canônico de segmento: `{"id": int, "start": float, "end": float, "text": str}`.
- Backends pesados (NeMo, faster-whisper, Chatterbox) sempre com **import preguiçoso** dentro do método.
- Testes unitários **não podem depender de GPU nem de rede**.
- Preservar os comportamentos existentes: `ctx.reporter` (spinner/bar), `free_cuda()` após estágios de GPU, resume por clipe já presente, escrita atômica (`_write_wav_atomic`), e o manifesto `05_clips.json` com `sample_rate`.
- Não adicionar comentários novos ao código (os comentários existentes permanecem).
- Commits em `feat:` / `test:` / `chore:`, um por tarefa.

---

### Task P1: Seam de ASR + normalizador + campos de config de ASR

**Files:**
- Create: `vpm2/asr/__init__.py` (empty)
- Create: `vpm2/asr/base.py`
- Create: `vpm2/asr/normalize.py`
- Modify: `vpm2/config.py`
- Create: `tests/test_normalize.py`
- Create: `tests/test_asr_base.py`

**Interfaces:**
- Produces:
  - `vpm2.asr.base.RawSegment(start: float, end: float, text: str)` (dataclass frozen)
  - `vpm2.asr.base.ASRBackend` com `transcribe(audio_path: Path) -> list[RawSegment]`
  - `vpm2.asr.base.get_asr_backend(config) -> ASRBackend`
  - `vpm2.asr.normalize.normalize_segments(raw) -> list[dict]` → `{id,start,end,text}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_normalize.py`:

```python
from vpm2.asr.base import RawSegment
from vpm2.asr.normalize import normalize_segments


def test_assigns_sequential_ids_and_preserves_order():
    raw = [RawSegment(0.0, 1.0, "hello"), RawSegment(1.5, 2.0, "world")]
    assert normalize_segments(raw) == [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hello"},
        {"id": 1, "start": 1.5, "end": 2.0, "text": "world"},
    ]


def test_drops_empty_and_whitespace_only_segments():
    raw = [RawSegment(0.0, 1.0, "  "), RawSegment(1.0, 2.0, "ok"), RawSegment(2.0, 3.0, "")]
    out = normalize_segments(raw)
    assert [s["text"] for s in out] == ["ok"]
    assert [s["id"] for s in out] == [0]


def test_sorts_by_start_time():
    raw = [RawSegment(5.0, 6.0, "b"), RawSegment(1.0, 2.0, "a")]
    assert [s["text"] for s in normalize_segments(raw)] == ["a", "b"]
    assert [s["id"] for s in normalize_segments(raw)] == [0, 1]


def test_strips_text():
    raw = [RawSegment(0.0, 1.0, "  hi  ")]
    assert normalize_segments(raw)[0]["text"] == "hi"


def test_empty_input_returns_empty_list():
    assert normalize_segments([]) == []
```

Create `tests/test_asr_base.py`:

```python
import pytest

from vpm2.asr.base import get_asr_backend
from vpm2.config import Config


def test_unknown_backend_raises_without_importing_models():
    with pytest.raises(ValueError, match="unknown asr backend"):
        get_asr_backend(Config(asr_backend="nope"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_normalize.py tests/test_asr_base.py -q'`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpm2.asr'`

- [ ] **Step 3: Implement the seam**

Create `vpm2/asr/__init__.py` (empty file).

Create `vpm2/asr/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from vpm2.config import Config


@dataclass(frozen=True)
class RawSegment:
    start: float
    end: float
    text: str


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> list[RawSegment]:
        ...


def get_asr_backend(config: Config) -> ASRBackend:
    if config.asr_backend == "parakeet":
        from vpm2.asr.parakeet_backend import ParakeetBackend
        return ParakeetBackend(config)
    if config.asr_backend == "whisper":
        from vpm2.asr.whisper_backend import WhisperBackend
        return WhisperBackend(config)
    raise ValueError(f"unknown asr backend: {config.asr_backend}")
```

Create `vpm2/asr/normalize.py`:

```python
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
```

- [ ] **Step 4: Add the ASR config fields**

In `vpm2/config.py`, add three fields at the TOP of the dataclass (before `asr_model`), leaving every existing field and comment intact:

```python
    asr_backend: str = "parakeet"          # "parakeet" | "whisper"
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    asr_model: str = "large-v3"            # Whisper model (used by the whisper backend)
```

Remove the old `asr_model: str = "large-v3"` line that was already there (keep only the one above).

- [ ] **Step 5: Run tests to verify they pass**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_normalize.py tests/test_asr_base.py -q'`
Expected: PASS (6 tests)

- [ ] **Step 6: Full suite**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest -q'`
Expected: PASS (the existing `test_cli.py` / `test_synthesize.py` / `test_translate.py` must still pass; `Config` keeps all prior fields).

- [ ] **Step 7: Commit**

```bash
wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && git add vpm2/asr/ vpm2/config.py tests/test_normalize.py tests/test_asr_base.py && git commit -m "feat: add pluggable ASR backend seam and segment normalizer"'
```

---

### Task P2: Backends Whisper + Parakeet e integração no TranscribeStage

**Files:**
- Create: `vpm2/asr/whisper_backend.py`
- Create: `vpm2/asr/parakeet_backend.py`
- Modify: `vpm2/stages/transcribe.py`
- Create: `tests/test_transcribe_stage.py`
- Create: `tests/test_parakeet_parse.py`
- Modify: `tests/test_asr_base.py`

**Interfaces:**
- Consumes: `vpm2.asr.base.RawSegment`, `get_asr_backend`, `vpm2.asr.normalize.normalize_segments`, `vpm2.gpu.free_cuda`, `vpm2.artifacts.valid_transcript/write_json`
- Produces: `WhisperBackend`, `ParakeetBackend`, `parse_hypothesis(hyp, audio_duration=None)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_asr_base.py`:

```python
def test_whisper_backend_selected_by_config():
    assert type(get_asr_backend(Config(asr_backend="whisper"))).__name__ == "WhisperBackend"


def test_parakeet_is_default_backend():
    assert type(get_asr_backend(Config())).__name__ == "ParakeetBackend"
```

Create `tests/test_parakeet_parse.py`:

```python
from vpm2.asr.base import RawSegment
from vpm2.asr.parakeet_backend import parse_hypothesis


class FakeHyp:
    def __init__(self, text, timestamp=None):
        self.text = text
        self.timestamp = timestamp


def test_parses_segment_timestamps():
    hyp = FakeHyp("hello world", {"segment": [
        {"segment": "hello", "start": 0.0, "end": 0.5},
        {"segment": "world", "start": 0.5, "end": 1.0},
    ]})
    assert parse_hypothesis(hyp) == [
        RawSegment(0.0, 0.5, "hello"),
        RawSegment(0.5, 1.0, "world"),
    ]


def test_falls_back_to_single_segment_with_audio_duration():
    assert parse_hypothesis(FakeHyp("no timing here"), audio_duration=12.5) == [
        RawSegment(0.0, 12.5, "no timing here"),
    ]


def test_empty_text_yields_no_segments():
    assert parse_hypothesis(FakeHyp("   "), audio_duration=3.0) == []
```

Create `tests/test_transcribe_stage.py`:

```python
from vpm2.artifacts import read_json
from vpm2.asr.base import RawSegment
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import transcribe as transcribe_mod
from vpm2.stages.transcribe import TranscribeStage


class FakeASRBackend:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path):
        return self._segments


def make_ctx(tmp_path):
    return Context(url="x", work_dir=tmp_path, config=Config())


def test_stage_writes_normalized_transcript(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    fake = FakeASRBackend([
        RawSegment(2.0, 3.0, "b"),
        RawSegment(0.0, 1.0, "a"),
        RawSegment(1.0, 2.0, "   "),
    ])
    monkeypatch.setattr(transcribe_mod, "get_asr_backend", lambda cfg: fake)

    TranscribeStage().run(ctx)

    data = read_json(tmp_path / "03_transcript.json")
    assert data["language"] == "en"
    assert [s["text"] for s in data["segments"]] == ["a", "b"]
    assert [s["id"] for s in data["segments"]] == [0, 1]


def test_is_done_false_when_missing(tmp_path):
    assert TranscribeStage().is_done(make_ctx(tmp_path)) is False


def test_is_done_true_after_run(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(
        transcribe_mod, "get_asr_backend",
        lambda cfg: FakeASRBackend([RawSegment(0.0, 1.0, "a")]),
    )
    stage = TranscribeStage()
    stage.run(ctx)
    assert stage.is_done(ctx) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_transcribe_stage.py tests/test_parakeet_parse.py tests/test_asr_base.py -q'`
Expected: FAIL — missing backend modules / transcribe not integrated.

- [ ] **Step 3: Implement the backends**

Create `vpm2/asr/whisper_backend.py`:

```python
from pathlib import Path

from vpm2.asr.base import ASRBackend, RawSegment
from vpm2.config import Config


class WhisperBackend(ASRBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._config.asr_model, device="cuda", compute_type="float16",
            )

    def transcribe(self, audio_path: Path) -> list[RawSegment]:
        self._ensure_model()
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self._config.source_lang,
            vad_filter=True,
        )
        return [
            RawSegment(float(s.start), float(s.end), s.text)
            for s in segments
        ]
```

Create `vpm2/asr/parakeet_backend.py`:

```python
from pathlib import Path

from vpm2.asr.base import ASRBackend, RawSegment
from vpm2.config import Config


def parse_hypothesis(hyp, audio_duration: float | None = None) -> list[RawSegment]:
    timestamp = getattr(hyp, "timestamp", None) or {}
    items = timestamp.get("segment") or []
    if items:
        return [
            RawSegment(float(t["start"]), float(t["end"]), str(t["segment"]))
            for t in items
        ]
    text = (getattr(hyp, "text", "") or "").strip()
    if not text:
        return []
    return [RawSegment(0.0, float(audio_duration or 0.0), text)]


class ParakeetBackend(ASRBackend):
    def __init__(self, config: Config):
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import nemo.collections.asr as nemo_asr
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._config.parakeet_model,
            )
            self._model.eval()

    def transcribe(self, audio_path: Path) -> list[RawSegment]:
        import soundfile as sf

        self._ensure_model()
        results = self._model.transcribe([str(audio_path)], timestamps=True)
        if not results:
            return []
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate
        return parse_hypothesis(results[0], audio_duration=duration)
```

- [ ] **Step 4: Integrate the transcribe stage**

Replace the contents of `vpm2/stages/transcribe.py` with:

```python
from pathlib import Path

from vpm2.artifacts import valid_transcript, write_json
from vpm2.asr.base import get_asr_backend
from vpm2.asr.normalize import normalize_segments
from vpm2.context import Context
from vpm2.gpu import free_cuda
from vpm2.stages.base import Stage


class TranscribeStage(Stage):
    name = "transcribe"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("03_transcript.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_transcript(self.output_path(ctx))

    def run(self, ctx: Context) -> None:
        backend = get_asr_backend(ctx.config)
        with ctx.reporter.spinner(f"transcrevendo ({ctx.config.asr_backend})"):
            raw = backend.transcribe(ctx.path("02_audio.wav"))
        segments = normalize_segments(raw)
        write_json(self.output_path(ctx), {
            "language": ctx.config.source_lang,
            "segments": segments,
        })
        del backend
        free_cuda()
```

> Nota de comportamento: o progresso fino por segmento do Whisper é substituído por um spinner (o backend esconde a iteratividade). O resultado (`03_transcript.json`) é idêntico.

- [ ] **Step 5: Run tests to verify they pass**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_transcribe_stage.py tests/test_parakeet_parse.py tests/test_asr_base.py -q'`
Expected: PASS (9 tests)

- [ ] **Step 6: Full suite**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest -q'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && git add vpm2/asr/ vpm2/stages/transcribe.py tests/test_transcribe_stage.py tests/test_parakeet_parse.py tests/test_asr_base.py && git commit -m "feat: add Parakeet and Whisper ASR backends"'
```

---

### Task P3: Dependência NeMo + verificação em GPU (checkpoint de risco)

**Files:**
- Create: `scripts/check_parakeet.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `vpm2.asr.parakeet_backend.ParakeetBackend`, `vpm2.config.Config`

> **Checkpoint:** se o NeMo não instalar/carregar no ambiente WSL (cu128/Blackwell), **pare** e mantenha `asr_backend="whisper"` como default em `vpm2/config.py`, adaptando `tests/test_asr_base.py::test_parakeet_is_default_backend` para afirmar o default `"whisper"`. As Tasks P4–P6 não dependem do Parakeet.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"nemo_toolkit[asr]"` to `dependencies`, after `"faster-whisper",`:

```toml
    "faster-whisper",
    "nemo_toolkit[asr]",
    "chatterbox-tts>=0.1.7",
```

Then:

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv sync'`
Expected: resolve/instala. Pode ser demorado — use timeout generoso (900000 ms). Se falhar por conflito de `torch`, o override existente (`torch>=2.7.0`) já cobre.

- [ ] **Step 2: Create the verification script**

Create `scripts/check_parakeet.py`:

```python
import sys
from pathlib import Path

from vpm2.asr.parakeet_backend import ParakeetBackend
from vpm2.config import Config


def main() -> None:
    audio = Path(sys.argv[1])
    segments = ParakeetBackend(Config()).transcribe(audio)
    for s in segments:
        print(f"[{s.start:7.2f} - {s.end:7.2f}] {s.text}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify runtime (GPU)**

Run import:
`wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run python -c "import nemo.collections.asr; print(\"nemo ok\")"'`
Expected: imprime `nemo ok`.

Then a smoke test on silence (verifies model load + VRAM, no speech needed):
`wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run python -c "import numpy as np, soundfile as sf; sf.write(\"/tmp/sil.wav\", np.zeros(32000, dtype=\"float32\"), 16000)" && uv run python scripts/check_parakeet.py /tmp/sil.wav'`
Expected: carrega o modelo e não crasha (silêncio pode não imprimir segmentos).

If any `work/*/02_audio.wav` exists, also run `uv run python scripts/check_parakeet.py <arquivo>` and note the printed English segments.

- [ ] **Step 4: Full suite**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest -q'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && git add pyproject.toml uv.lock scripts/check_parakeet.py && git commit -m "feat: add NeMo dependency and Parakeet verification script"'
```

Se cair no branch de falha: em vez disso, ajuste o default para `"whisper"` + o teste, e comite `chore: default ASR to whisper (NeMo unavailable)`.

---

### Task P4: Módulo puro de perfis de voz

**Files:**
- Create: `vpm2/voices/__init__.py` (empty)
- Create: `vpm2/voices/profile.py`
- Create: `tests/test_voice_profile.py`

**Interfaces:**
- Produces: `VoiceProfile`, `profile_dir`, `sha256_file`, `make_profile`, `save_profile`, `load_profile`, `resolve_reference`, `cache_key`, `transcript_for_window`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice_profile.py`:

```python
import pytest

from vpm2.voices import profile as vp


def _write_bytes(path, content=b"RIFFfake"):
    path.write_bytes(content)
    return path


def test_make_profile_captures_sha_and_span(tmp_path):
    ref = _write_bytes(tmp_path / "ref.wav", b"abc")
    p = vp.make_profile("bob", ref, source_video="vid1", span=(1.0, 11.0),
                        transcript="hi", backend="chatterbox")
    assert p.id == "bob"
    assert p.span_start == 1.0 and p.span_end == 11.0
    assert p.sha256 == vp.sha256_file(ref)


def test_save_and_load_roundtrip(tmp_path):
    ref = _write_bytes(tmp_path / "ref.wav")
    p = vp.make_profile("bob", ref, source_video="v", span=(0.0, 7.0),
                        transcript="t", backend="chatterbox")
    vp.save_profile(tmp_path, p)
    assert vp.load_profile(tmp_path, "bob") == p


def test_resolve_reference_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        vp.resolve_reference(tmp_path, "nope")


def test_cache_key_stable_and_sensitive():
    a = vp.cache_key("sha1", "hello", {"backend": "chatterbox"})
    b = vp.cache_key("sha1", "hello", {"backend": "chatterbox"})
    c = vp.cache_key("sha1", "hello", {"backend": "other"})
    d = vp.cache_key("sha1", "HELLO", {"backend": "chatterbox"})
    assert a == b
    assert a != c
    assert a != d


def test_transcript_for_window_joins_overlapping_segments():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "one"},
        {"start": 2.0, "end": 4.0, "text": "two"},
        {"start": 10.0, "end": 12.0, "text": "far"},
    ]
    assert vp.transcript_for_window(segs, (1.0, 3.0)) == "one two"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_voice_profile.py -q'`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpm2.voices'`

- [ ] **Step 3: Implement**

Create `vpm2/voices/__init__.py` (empty file).

Create `vpm2/voices/profile.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_voice_profile.py -q'`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && git add vpm2/voices/ tests/test_voice_profile.py && git commit -m "feat: add voice profile module"'
```

---

### Task P5: Sintetize — modo `profile`, promoção a perfil e cache de TTS

**Files:**
- Modify: `vpm2/config.py` (add `voice_profile`, `voices_dir`, `save_profile`, `tts_cache`, `tts_cache_dir`)
- Modify: `vpm2/stages/synthesize.py`
- Modify: `tests/test_synthesize.py` (desligar cache no `make_ctx` existente)
- Create: `tests/test_synthesize_cache.py`
- Create: `tests/test_synthesize_profiles.py`

**Interfaces:**
- Consumes: `vpm2.voices.profile` (todo), `vpm2.context.Context`, `vpm2.tts.base.get_backend`, `vpm2.voice_sample.pick_reference_window`
- Produces: `SynthesizeStage` com modos `cloning|preset|profile`, promoção a perfil e cache entre vídeos.

- [ ] **Step 1: Update config fields**

In `vpm2/config.py`, update `voice_mode` and add the new fields right after it:

```python
    voice_mode: str = "cloning"         # "cloning" | "preset" | "profile"
    voice_profile: str | None = None    # required when voice_mode == "profile"
    voices_dir: str = "voices"
    save_profile: str | None = None     # when set + cloning, persist the extracted reference
    tts_cache: bool = True              # reuse clips across videos by hash(profile, text)
    tts_cache_dir: str = "work/_tts_cache"
```

Leave `preset_ref_wav` and all other fields as they are.

- [ ] **Step 2: Write the failing tests**

Modify `tests/test_synthesize.py`: change `make_ctx` so the existing tests do not touch the shared cache dir. Replace its `cfg = Config(...)` line with:

```python
    cfg = Config(voice_mode="preset", preset_ref_wav=str(ref), tts_cache=False)
```

Create `tests/test_synthesize_cache.py`:

```python
import numpy as np
import soundfile as sf

from vpm2.artifacts import write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import synthesize as synth_mod
from vpm2.stages.synthesize import SynthesizeStage

SR = 24000


class FakeBackend:
    sample_rate = SR

    def __init__(self):
        self.calls = []

    def synth(self, text, ref):
        self.calls.append(text)
        return np.zeros(SR // 2, dtype="float32")


def _ctx(tmp_path, name, cache_dir):
    ref = tmp_path / "ref.wav"
    if not ref.exists():
        sf.write(str(ref), np.zeros(SR, dtype="float32"), SR)
    cfg = Config(voice_mode="preset", preset_ref_wav=str(ref),
                 tts_cache=True, tts_cache_dir=str(cache_dir))
    ctx = Context(url="x", work_dir=tmp_path / name, config=cfg)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(ctx.path("04_translation.json"), {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "en", "text_pt": "pt"}]})
    return ctx


def test_cache_reused_across_videos(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    fake = FakeBackend()
    monkeypatch.setattr(synth_mod, "get_backend", lambda cfg: fake)

    SynthesizeStage().run(_ctx(tmp_path, "v1", cache))
    SynthesizeStage().run(_ctx(tmp_path, "v2", cache))

    assert fake.calls == ["pt"]
```

Create `tests/test_synthesize_profiles.py`:

```python
import numpy as np
import pytest
import soundfile as sf

from vpm2.artifacts import write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import synthesize as synth_mod
from vpm2.stages.synthesize import SynthesizeStage
from vpm2.voices import profile as vp

SR = 24000


class FakeBackend:
    sample_rate = SR

    def __init__(self):
        self.refs = []

    def synth(self, text, ref):
        self.refs.append(ref)
        return np.zeros(SR // 2, dtype="float32")


def _ctx(tmp_path, **cfg):
    ctx = Context(url="vid", work_dir=tmp_path / "work", config=Config(**cfg))
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(ctx.path("04_translation.json"), {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "en", "text_pt": "pt"}]})
    return ctx


def test_profile_mode_uses_saved_reference(tmp_path, monkeypatch):
    voices = tmp_path / "voices"
    ref = voices / "bob" / "ref.wav"
    ref.parent.mkdir(parents=True)
    sf.write(str(ref), np.zeros(SR, dtype="float32"), SR)
    p = vp.make_profile("bob", ref, source_video="v", span=(0.0, 1.0),
                        transcript="hi", backend="chatterbox")
    vp.save_profile(voices, p)

    ctx = _ctx(tmp_path, voice_mode="profile", voice_profile="bob",
               voices_dir=str(voices), tts_cache_dir=str(tmp_path / "cache"))
    fake = FakeBackend()
    monkeypatch.setattr(synth_mod, "get_backend", lambda cfg: fake)

    SynthesizeStage().run(ctx)
    assert fake.refs == [ref]


def test_profile_mode_without_id_raises(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, voice_mode="profile",
               tts_cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(synth_mod, "get_backend", lambda cfg: FakeBackend())
    with pytest.raises(ValueError):
        SynthesizeStage().run(ctx)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_synthesize_cache.py tests/test_synthesize_profiles.py -q'`
Expected: FAIL (feature "profile"/cache not present)

- [ ] **Step 4: Rewrite the synthesize stage**

Replace the contents of `vpm2/stages/synthesize.py` with:

```python
import os
import shutil
from pathlib import Path

import soundfile as sf

from vpm2.artifacts import read_json, valid_clips, write_json
from vpm2.context import Context
from vpm2.gpu import free_cuda
from vpm2.stages.base import Stage
from vpm2.tts.base import get_backend
from vpm2.voice_sample import pick_reference_window
from vpm2.voices import profile as vp


def _write_wav_atomic(dest: Path, audio, sr: int) -> None:
    # Write to a sibling temp file then rename: os.replace is atomic on the same
    # filesystem, so an interrupted run can never leave a half-written clip that
    # a later resume would mistake for finished work.
    tmp = dest.with_name(f".{dest.name}.tmp")
    # explicit format: the temp name's .tmp suffix hides the wav extension that
    # soundfile would otherwise infer.
    sf.write(str(tmp), audio, sr, format="WAV")
    os.replace(tmp, dest)


def _extract_reference(ctx: Context):
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    audio_path = ctx.path("02_audio.wav")
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    # faster-whisper VAD expects 16kHz float32; timestamps come back in SAMPLES.
    ts = get_speech_timestamps(
        data.astype("float32"),
        vad_options=VadOptions(),
        sampling_rate=sr,
    )
    spans = [(t["start"] / sr, t["end"] / sr) for t in ts]
    win = pick_reference_window(spans)
    ref_path = ctx.path("ref_voice.wav")
    if win is None:
        # fallback: first 10s
        start, end = 0.0, min(10.0, len(data) / sr)
    else:
        start, end = win
    sf.write(str(ref_path), data[int(start * sr):int(end * sr)], sr)
    return ref_path, (float(start), float(end))


class SynthesizeStage(Stage):
    name = "synthesize"

    def output_path(self, ctx: Context) -> Path:
        return ctx.path("05_clips.json")

    def is_done(self, ctx: Context) -> bool:
        return valid_clips(self.output_path(ctx), ctx.path("05_clips"))

    def _resolve_reference(self, ctx: Context):
        cfg = ctx.config
        if cfg.voice_mode == "cloning":
            with ctx.reporter.spinner("extraindo voz de referência do vídeo"):
                ref, win = _extract_reference(ctx)
            sha = vp.sha256_file(ref)
            if cfg.save_profile:
                segments = read_json(ctx.path("03_transcript.json"))["segments"]
                transcript = vp.transcript_for_window(segments, win)
                profile = vp.make_profile(
                    cfg.save_profile, ref, source_video=ctx.url, span=win,
                    transcript=transcript, backend=cfg.tts_backend,
                )
                vp.save_profile(cfg.voices_dir, profile)
                shutil.copyfile(
                    ref, vp.profile_dir(cfg.voices_dir, cfg.save_profile) / "ref.wav"
                )
                sha = profile.sha256
            return ref, sha
        if cfg.voice_mode == "preset":
            if not cfg.preset_ref_wav:
                raise ValueError(
                    "voice_mode='preset' requires a reference clip. "
                    "Pass --preset-ref <clean_pt_voice.wav> "
                    "(Chatterbox has no built-in preset voices)."
                )
            ref = Path(cfg.preset_ref_wav)
            if not ref.exists():
                raise FileNotFoundError(f"preset_ref_wav not found: {ref}")
            return ref, vp.sha256_file(ref)
        if cfg.voice_mode == "profile":
            if not cfg.voice_profile:
                raise ValueError("voice_mode='profile' requires voice_profile")
            profile = vp.load_profile(cfg.voices_dir, cfg.voice_profile)
            return vp.resolve_reference(cfg.voices_dir, cfg.voice_profile), profile.sha256
        raise ValueError(f"unknown voice_mode: {cfg.voice_mode}")

    def run(self, ctx: Context) -> None:
        clips_dir = ctx.path("05_clips")
        clips_dir.mkdir(parents=True, exist_ok=True)
        segs = read_json(ctx.path("04_translation.json"))["segments"]

        def clip_path(seg) -> Path:
            return clips_dir / f"{seg['id']:04d}.wav"

        # Resume: a clip already on disk is complete (clips are written
        # atomically), so reuse it. Only spin up the reference + TTS model when
        # something is actually missing -- a full resume after a crash that only
        # lost the manifest costs no model load and no GPU.
        pending = [s for s in segs if not clip_path(s).exists()]
        backend = ref = profile_sha = None
        sample_rate = None
        cache_dir = None
        if pending:
            ref, profile_sha = self._resolve_reference(ctx)
            with ctx.reporter.spinner("carregando modelo de voz (Chatterbox TTS)"):
                backend = get_backend(ctx.config)
            sample_rate = backend.sample_rate
            if ctx.config.tts_cache:
                cache_dir = Path(ctx.config.tts_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

        out = []
        with ctx.reporter.bar("sintetizando voz PT-BR", total=len(segs)) as bar:
            for s in segs:
                dest = clip_path(s)
                if dest.exists():
                    info = sf.info(str(dest))
                    if sample_rate is None:
                        sample_rate = info.samplerate
                    duration = info.frames / info.samplerate
                else:
                    cached = None
                    if cache_dir is not None:
                        key = vp.cache_key(profile_sha, s["text_pt"], {
                            "backend": ctx.config.tts_backend,
                            "target_lang": ctx.config.target_lang,
                            "sample_rate": sample_rate,
                        })
                        cached = cache_dir / f"{key}.wav"
                    if cached is not None and cached.exists():
                        shutil.copyfile(cached, dest)
                        info = sf.info(str(dest))
                        duration = info.frames / info.samplerate
                    else:
                        audio = backend.synth(s["text_pt"], ref)
                        _write_wav_atomic(dest, audio, backend.sample_rate)
                        if cached is not None:
                            _write_wav_atomic(cached, audio, backend.sample_rate)
                        duration = len(audio) / backend.sample_rate
                out.append({
                    "id": s["id"], "start": s["start"], "end": s["end"],
                    "clip": dest.name, "duration": duration,
                })
                bar.advance()
        if backend is not None:
            del backend
            free_cuda()
        write_json(self.output_path(ctx), {
            "sample_rate": sample_rate, "segments": out,
        })
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_synthesize.py tests/test_synthesize_cache.py tests/test_synthesize_profiles.py -q'`
Expected: PASS (7 tests: os 3 antigos + 1 de cache + 2 de perfis)

- [ ] **Step 6: Full suite**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest -q'`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && git add vpm2/config.py vpm2/stages/synthesize.py tests/test_synthesize.py tests/test_synthesize_cache.py tests/test_synthesize_profiles.py && git commit -m "feat: add voice profile mode and cross-video TTS cache"`
```

---

### Task P6: Flags de CLI

**Files:**
- Modify: `vpm2/cli.py`
- Test: `tests/test_cli.py` (verificar que continua passando)

**Interfaces:**
- Consumes: `vpm2.config.Config`
- Produces: flags `--asr-backend`, `--voice-profile`, `--save-profile`.

- [ ] **Step 1: Read the current CLI test**

Run: `wsl -d Ubuntu -- cat /home/pvs/projetos/vpm2/tests/test_cli.py`
Expected: entenda o que o teste cobre (parsing/args) antes de mexer, para não quebrar.

- [ ] **Step 2: Add the flags**

In `vpm2/cli.py`, inside `main`, update the `--voice-mode` choices and add new arguments next to the existing ones:

```python
    ap.add_argument("--voice-mode", choices=["cloning", "preset", "profile"],
                    default="cloning",
                    help="cloning = auto-extract reference from the video; "
                         "preset = use --preset-ref clip; "
                         "profile = reuse a saved voice profile (--voice-profile)")
    ap.add_argument("--voice-profile", default=None,
                    help="saved voice profile id (required when --voice-mode profile)")
    ap.add_argument("--save-profile", default=None,
                    help="persist the auto-extracted reference under this profile id")
    ap.add_argument("--asr-backend", choices=["parakeet", "whisper"],
                    default="parakeet")
```

Then, after `config = Config(...)`, add validation and wiring:

```python
    if args.voice_mode == "profile" and not args.voice_profile:
        ap.error("--voice-mode profile requires --voice-profile <id>")
    config.voice_profile = args.voice_profile
    config.save_profile = args.save_profile
    config.asr_backend = args.asr_backend
```

(Keep the existing `--asr-model` flag as-is; it now selects the Whisper model.)

- [ ] **Step 3: Run the CLI test + full suite**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run pytest tests/test_cli.py -q && uv run pytest -q'`
Expected: PASS

- [ ] **Step 4: Sanity-check the help text**

Run: `wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && uv run vpm2 --help'`
Expected: mostra `--asr-backend`, `--voice-profile`, `--save-profile`, e `profile` entre os `--voice-mode`.

- [ ] **Step 5: Commit**

```bash
wsl -d Ubuntu -- bash -lc 'cd /home/pvs/projetos/vpm2 && git add vpm2/cli.py && git commit -m "feat: expose ASR backend and voice profile flags in CLI"'
```

---

## Self-Review

**Spec coverage:**
- ASR plugável (Parakeet padrão, Whisper fallback) → P1–P3.
- Perfis de voz persistidos (`voices/<id>/`) → P4; modo `profile` + promoção → P5.
- Cache de TTS entre vídeos (`hash(perfil, texto, params)`) → P5.
- Preservar reporter/free_cuda/resume/atomicidade → P2 e P5.
- Exposição via CLI → P6.

**Placeholder scan:** sem "TBD/TODO"; todos os steps trazem código e comandos completos.

**Type consistency:** `RawSegment(start,end,text)`; `normalize_segments -> {id,start,end,text}`; `VoiceProfile` com `sha256`/`span_start`/`span_end`; `cache_key(profile_sha, text, params)`; `_resolve_reference -> (ref, profile_sha)`; manifesto `05_clips.json` inalterado + cache em `tts_cache_dir`.

**Gap aceito:** a chave de cache usa `sample_rate` lido antes do primeiro `synth` (default da classe quando o backend ainda não carregou); é consistente entre execuções.
