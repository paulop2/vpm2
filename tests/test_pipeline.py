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


def test_force_from_unknown_stage_raises(tmp_path):
    import pytest
    a = FakeStage("a", done=True)
    with pytest.raises(ValueError):
        run_pipeline(make_ctx(tmp_path), stages=[a], force_from="nope")