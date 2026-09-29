import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vpm2.artifacts import write_json
from vpm2.config import Config
from vpm2.context import Context
from vpm2.stages import assemble as assemble_mod
from vpm2.stages.assemble import AssembleStage

SR = 24000


def make_ctx(tmp_path):
    """Minimal work dir with a valid 05 artifact and one short clip.

    The single clip (0.5s) sits inside a 10s video and never overruns, so
    plan_timeline keeps speed=1: the only subprocess call is the mux.
    """
    ctx = Context(url="x", work_dir=tmp_path, config=Config())
    write_json(ctx.path("01_meta.json"), {"duration": 10.0})
    clips_dir = ctx.path("05_clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(clips_dir / "0000.wav"),
             np.zeros(SR // 2, dtype="float32"), SR)
    write_json(ctx.path("05_clips.json"), {
        "sample_rate": SR,
        "segments": [{"id": 0, "start": 0.0, "end": 1.0,
                      "clip": "0000.wav", "duration": 0.5}],
    })
    return ctx


def test_success_writes_final_and_leaves_no_temp(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)

    written = []

    def fake_run(cmd, *args, **kwargs):
        # the mux command writes to the last path argument
        written.append(cmd[-1])
        Path(cmd[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(assemble_mod.subprocess, "run", fake_run)

    stage = AssembleStage()
    stage.run(ctx)

    # ffmpeg must target a temp sibling, never the final file directly
    assert written == [str(tmp_path / ".06_final.tmp.mp4")]
    assert ctx.path("06_final.mp4").exists()
    assert ctx.path("06_audio_pt.wav").exists()
    assert stage.is_done(ctx) is True
    # atomic writes must not leave partial artifacts behind
    assert not (tmp_path / ".06_final.tmp.mp4").exists()
    assert not (tmp_path / ".06_audio_pt.tmp.wav").exists()


def test_mux_failure_leaves_no_final_and_is_not_done(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)

    def fake_run(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(assemble_mod.subprocess, "run", fake_run)

    stage = AssembleStage()
    with pytest.raises(subprocess.CalledProcessError):
        stage.run(ctx)

    assert not ctx.path("06_final.mp4").exists()
    assert stage.is_done(ctx) is False
    # no partial mux output left behind either
    assert not (tmp_path / ".06_final.tmp.mp4").exists()


def test_interrupted_mux_partial_file_is_not_treated_as_done(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)

    def fake_run(cmd, *args, **kwargs):
        # simulate ffmpeg dying mid-write: bytes hit the target, then it fails
        Path(cmd[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(assemble_mod.subprocess, "run", fake_run)

    stage = AssembleStage()
    with pytest.raises(subprocess.CalledProcessError):
        stage.run(ctx)

    assert not ctx.path("06_final.mp4").exists()
    assert stage.is_done(ctx) is False
    assert not (tmp_path / ".06_final.tmp.mp4").exists()
