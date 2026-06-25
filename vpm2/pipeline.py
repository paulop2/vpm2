from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.stages.download import DownloadStage
from vpm2.stages.extract_audio import ExtractAudioStage
from vpm2.stages.transcribe import TranscribeStage
from vpm2.stages.translate import TranslateStage
from vpm2.stages.synthesize import SynthesizeStage
from vpm2.stages.assemble import AssembleStage

# Concrete stages are appended in their own tasks to keep imports light here.
STAGES: list[Stage] = [
    DownloadStage(),
    ExtractAudioStage(),
    TranscribeStage(),
    TranslateStage(),
    SynthesizeStage(),
    AssembleStage(),
]


def run_pipeline(ctx: Context, stages: list[Stage] | None = None,
                 force_from: str | None = None, reporter=None) -> list[str]:
    stages = STAGES if stages is None else stages
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    if reporter is not None:
        ctx.reporter = reporter
    reporter = ctx.reporter

    if force_from is not None and force_from not in {s.name for s in stages}:
        raise ValueError(
            f"--force: unknown stage '{force_from}'. "
            f"Valid stages: {[s.name for s in stages]}"
        )

    total = len(stages)
    forcing = False
    executed: list[str] = []
    for i, stage in enumerate(stages, start=1):
        if force_from is not None and stage.name == force_from:
            forcing = True
        if forcing or not stage.is_done(ctx):
            reporter.stage_banner(i, total, stage.name)
            stage.run(ctx)
            executed.append(stage.name)
        else:
            reporter.stage_banner(i, total, stage.name, skipped=True)
    return executed