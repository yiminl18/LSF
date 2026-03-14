# -*- coding: utf-8 -*-
"""
Generic multiprocessing Pool + rich progress bar orchestration utility.

Extracts the ~70 lines of nearly identical Pool+Queue+progress boilerplate
from train_model and evaluate_model into a reusable function.
"""

import time
import queue as _queue_mod
import multiprocessing
from typing import Any, Callable, Dict, Optional

from core.utils.progress import create_pipeline_progress


def run_pool_with_progress(
    build_tasks: Callable,
    worker_fn: Callable,
    workers: int,
    *,
    initializer: Optional[Callable] = None,
    initargs: tuple = (),
    maxtasksperchild: int = 4,
    task_id_index: int = 1,
    description: str = "Total Progress",
    result_collector: Optional[Callable] = None,
    start_method: Optional[str] = None,
) -> Dict[Any, Any]:
    """
    Generic multiprocessing Pool + rich progress bar + Queue orchestration.

    Args:
        build_tasks: Callback that receives a progress_queue and returns a list of task tuples.
                     The caller embeds the queue into each task tuple within this function.
        worker_fn: Worker function with signature worker_fn(task_tuple) -> result.
        workers: Number of parallel processes.
        initializer: Pool initializer function.
        initargs: Pool initializer arguments.
        maxtasksperchild: Maximum tasks each child process handles.
        task_id_index: Index of the task ID within the task tuple.
        description: Progress bar description text.
        result_collector: Optional callback with signature (task_id, result) -> None.
        start_method: Multiprocessing start method (None=system default, or "spawn"/"forkserver"/"fork").

    Returns:
        {task_id: result} dictionary.

    Workers send progress messages via the Queue embedded in the task tuple:
      ("progress", task_id, current, total)
      ("log", task_id, text)
    """
    all_results: Dict[Any, Any] = {}
    ctx = multiprocessing.get_context(start_method) if start_method else multiprocessing

    with ctx.Manager() as manager:
        progress_queue = manager.Queue()
        tasks = build_tasks(progress_queue)

        pool_kwargs = dict(
            processes=workers,
            maxtasksperchild=maxtasksperchild,
        )
        if initializer is not None:
            pool_kwargs["initializer"] = initializer
            pool_kwargs["initargs"] = initargs

        with ctx.Pool(**pool_kwargs) as pool:
            async_results: Dict[Any, Any] = {}
            for t in tasks:
                tid = t[task_id_index]
                async_results[tid] = pool.apply_async(worker_fn, (t,))

            pool.close()

            with create_pipeline_progress() as progress:
                main_task = progress.add_task(f"[green]{description}", total=len(tasks))
                q_tasks: Dict[Any, Any] = {}
                completed: set = set()

                while len(completed) < len(tasks):
                    # Process progress messages
                    try:
                        while not progress_queue.empty():
                            msg = progress_queue.get_nowait()
                            msg_type = msg[0]

                            if msg_type == "progress":
                                tid, curr, tot = msg[1], msg[2], msg[3]
                                if tid not in q_tasks:
                                    q_tasks[tid] = progress.add_task(
                                        f"Q{tid}", total=tot
                                    )
                                progress.update(q_tasks[tid], completed=curr)

                            elif msg_type == "log":
                                tid, txt = msg[1], msg[2]
                                progress.console.print(f"[Q{tid}] {txt}")

                    except _queue_mod.Empty:
                        pass

                    # Check for completed tasks
                    for tid, res_obj in async_results.items():
                        if tid in completed:
                            continue
                        if res_obj.ready():
                            try:
                                result = res_obj.get(timeout=0.1)
                                all_results[tid] = result
                                if result_collector is not None:
                                    result_collector(tid, result)
                                # Print warning if result contains an error key
                                if isinstance(result, dict) and "error" in result:
                                    progress.console.print(
                                        f"[red]Q{tid}: {str(result['error'])[:200]}[/red]"
                                    )
                            except Exception as e:
                                all_results[tid] = {"error": str(e)}
                                if result_collector is not None:
                                    result_collector(tid, {"error": str(e)})
                                progress.console.print(f"[red]Q{tid}: {e}[/red]")

                            completed.add(tid)
                            progress.advance(main_task)
                            if tid in q_tasks:
                                progress.update(q_tasks[tid], visible=False)

                    time.sleep(0.1)

                # Drain remaining messages from the queue to avoid blocking Manager shutdown
                while not progress_queue.empty():
                    try:
                        progress_queue.get_nowait()
                    except _queue_mod.Empty:
                        break

            pool.join()

    return all_results
