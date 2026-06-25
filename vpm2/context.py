from dataclasses import dataclass, field
from pathlib import Path

from vpm2.config import Config
from vpm2.progress import NullReporter


@dataclass
class Context:
    url: str
    work_dir: Path
    config: Config
    # Silent by default; the CLI swaps in a RichReporter for visual progress.
    reporter: object = field(default_factory=NullReporter)

    def path(self, name: str) -> Path:
        return self.work_dir / name

    def log_dir(self) -> Path:
        d = self.work_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d
