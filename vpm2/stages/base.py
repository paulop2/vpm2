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