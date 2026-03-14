# -*- coding: utf-8 -*-
"""
Rich Progress Bar Utilities

Provides standardized progress bars using the Rich library to replace tqdm.
Supports nested progress bars: inner functions automatically reuse the outer Progress instance.
"""

import contextvars
from typing import Callable, Iterable, Optional, Any
from contextlib import contextmanager
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.text import Text

# Track the currently active Progress instance (thread-safe)
_active_progress: contextvars.ContextVar[Optional[Progress]] = contextvars.ContextVar(
    "_active_progress", default=None
)


def get_active_progress() -> Optional[Progress]:
    """Get the currently active Progress instance, or None if there is none."""
    return _active_progress.get()


def create_progress(transient: bool = False, **kwargs) -> Progress:
    """
    Creates a standardized Rich Progress object.

    Args:
        transient: If True, the progress bar will disappear when complete.
        **kwargs: Additional arguments passed to Progress constructor.

    Returns:
        A configured Progress instance.
    """
    # Check if custom columns are provided in kwargs, otherwise use defaults
    if "columns" in kwargs:
        return Progress(transient=transient, **kwargs)

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),  # Shows "50%"
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        transient=transient,
        **kwargs,
    )


def rich_tqdm(
    sequence: Iterable,
    desc: str = "",
    total: Optional[int] = None,
    leave: bool = True,
    **kwargs: Any,
) -> Iterable:
    """
    A drop-in replacement for tqdm using Rich with standardized styling.
    Supports nesting: if an active Progress exists, it reuses it and adds a subtask.

    Args:
        sequence: The iterable to track.
        desc: Description of the task.
        total: Total number of items (optional).
        leave: If True, keeps the progress bar after completion. (Default: True)
        **kwargs: Ignored arguments for compatibility with tqdm (e.g. ncols, unit).

    Returns:
        An iterator that yields items from the sequence.
    """
    active = get_active_progress()

    if active is not None:
        # Reuse outer Progress and add a subtask
        yield from active.track(sequence, total=total, description=desc)
    else:
        # No active Progress; create a new one
        progress = create_progress(transient=not leave)
        with progress:
            yield from progress.track(sequence, total=total, description=desc)


def create_pipeline_progress(show_count: bool = True) -> Progress:
    """
    Create a progress bar tailored for pipeline scripts.

    Args:
        show_count: Whether to show the {completed}/{total} counter.

    Returns:
        A configured Progress instance.
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ]
    if show_count:
        columns.extend(["•", TextColumn("{task.completed}/{task.total}")])

    return Progress(*columns)


class CostColumn(ProgressColumn):
    """Progress bar column that displays real-time API cost."""

    def __init__(self, cost_getter: Callable[[], float]):
        super().__init__()
        self._cost_getter = cost_getter

    def render(self, task) -> Text:
        cost = self._cost_getter()
        return Text(f"${cost:.4f}", style="cyan")


def create_pipeline_progress_with_cost(
    cost_getter: Callable[[], float], show_count: bool = True
) -> Progress:
    """
    Create a pipeline progress bar with real-time cost display.

    Args:
        cost_getter: Callback that returns the current cumulative cost.
        show_count: Whether to show the {completed}/{total} counter.

    Returns:
        A configured Progress instance (with $x.xxxx on the right side).
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ]
    if show_count:
        columns.extend(["•", TextColumn("{task.completed}/{task.total}")])
    columns.extend(["•", CostColumn(cost_getter)])

    return Progress(*columns)


@contextmanager
def managed_progress(progress: Progress):
    """
    Context manager that registers a Progress instance as active.
    Inner functions calling rich_tqdm will automatically reuse this Progress, enabling nested progress bars.

    Usage:
        with managed_progress(create_pipeline_progress()) as progress:
            task = progress.add_task("Outer", total=10)
            for i in range(10):
                inner_work()  # rich_tqdm calls inside will reuse progress
                progress.advance(task)
    """
    token = _active_progress.set(progress)
    try:
        with progress:
            yield progress
    finally:
        _active_progress.reset(token)
