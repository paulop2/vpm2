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
