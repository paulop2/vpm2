from vpm2.context import Context
from vpm2.stages.base import Stage
from vpm2.stages.download import DownloadStage

# Concrete stages are appended in their own tasks to keep imports light here.
STAGES: list[Stage] = [
    DownloadStage(),
]


def run_pipeline(ctx: Context, stages: list[Stage] | None = None,
                 force_from: str | None = None) -> list[str]:
    stages = STAGES if stages is None else stages
    ctx.work_dir.mkdir(parents=True, exist_ok=True)

    if force_from is not None and force_from not in {s.name for s in stages}:
        raise ValueError(
            f"--force: unknown stage '{force_from}'. "
            f"Valid stages: {[s.name for s in stages]}"
        )

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