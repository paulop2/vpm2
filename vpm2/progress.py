"""Visual progress reporting for the pipeline.

The pipeline runs stages sequentially, so only one stage is ever active at a
time. That lets each stage open and tear down its own short-lived Rich widget
(a bar or a spinner) without worrying about nested live displays -- the stage
banners are plain console prints in between.

A :class:`NullReporter` is the default on :class:`~vpm2.context.Context` so that
library use and the test-suite stay silent; the CLI swaps in a
:class:`RichReporter` to get the bars/spinners.
"""

from contextlib import contextmanager


class _NullTask:
    def advance(self, n: float = 1) -> None: ...
    def update(self, *, completed: float | None = None,
               total: float | None = None) -> None: ...


class NullReporter:
    """No-op reporter: safe default for tests and library callers."""

    def stage_banner(self, index: int, total: int, name: str,
                     skipped: bool = False) -> None: ...

    def info(self, message: str) -> None: ...

    @contextmanager
    def bar(self, label: str, total: float, show_count: bool = True):
        yield _NullTask()

    @contextmanager
    def spinner(self, label: str):
        yield


class _Task:
    """Handle to one row of a live Rich progress bar."""

    def __init__(self, progress, task_id):
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: float = 1) -> None:
        self._progress.advance(self._task_id, n)

    def update(self, *, completed: float | None = None,
               total: float | None = None) -> None:
        kw = {}
        if completed is not None:
            kw["completed"] = completed
        if total is not None:
            kw["total"] = total
        if kw:
            self._progress.update(self._task_id, **kw)


class RichReporter:
    """Render stage banners, progress bars and spinners with `rich`."""

    def __init__(self):
        from rich.console import Console

        self.console = Console()

    def stage_banner(self, index: int, total: int, name: str,
                     skipped: bool = False) -> None:
        tag = f"[dim]\\[{index}/{total}][/dim]"
        if skipped:
            self.console.print(f"{tag} [green]✔[/green] {name} "
                               f"[dim](cache, pulando)[/dim]")
        else:
            self.console.print(f"{tag} [bold cyan]{name}[/bold cyan]")

    def info(self, message: str) -> None:
        self.console.print(f"   [dim]{message}[/dim]")

    @contextmanager
    def bar(self, label: str, total: float, show_count: bool = True):
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn,
            TextColumn, TimeElapsedColumn, TimeRemainingColumn,
        )

        columns = [
            TextColumn("   [cyan]{task.description}[/cyan]"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
        ]
        if show_count:
            columns.append(MofNCompleteColumn())
        columns += [
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]restam[/dim]"),
            TimeRemainingColumn(),
        ]
        progress = Progress(*columns, console=self.console, transient=False)
        with progress:
            task_id = progress.add_task(label, total=total)
            yield _Task(progress, task_id)
            # snap to 100% on a clean exit so the final frame isn't stuck at 99%
            progress.update(task_id, completed=total)

    @contextmanager
    def spinner(self, label: str):
        with self.console.status(f"[cyan]{label}[/cyan]", spinner="dots"):
            yield
