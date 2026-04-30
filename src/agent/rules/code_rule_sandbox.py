"""Sandboxed execution of LLM-generated locate_region() Python code.

Safety layers:
  1. AST whitelist — reject code with forbidden constructs before execution.
  2. Restricted globals — only expose safe builtins + `re` module.
  3. Timeout — hard cap on wall-clock execution time.
  4. Return-value validation — must be a non-empty substring of the input.
"""

from __future__ import annotations

import ast
import builtins
import multiprocessing
import re
import time
from dataclasses import dataclass
from typing import Any


# ─── AST Validation ──────────────────────────────────────────────────────────

_ALLOWED_IMPORTS: frozenset[str] = frozenset({"re"})

_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "print",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "exit",
        "quit",
    }
)

_FORBIDDEN_ATTR_PREFIXES: tuple[str, ...] = ("__",)
_ALLOWED_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__init__",
        "__str__",
        "__repr__",
        "__len__",
        "__contains__",
        "__iter__",
        "__next__",
        "__enter__",
        "__exit__",
    }
)


def validate_code_ast(code: str) -> list[str]:
    """Parse *code* and return a list of safety violations (empty = OK)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    violations: list[str] = []

    for node in ast.walk(tree):
        # --- forbidden imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in _ALLOWED_IMPORTS:
                violations.append(f"forbidden import from: {mod}")

        # --- forbidden names ---
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.append(f"forbidden name: {node.id}")

        # --- forbidden attributes ---
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            for prefix in _FORBIDDEN_ATTR_PREFIXES:
                if attr.startswith(prefix) and attr not in _ALLOWED_DUNDER_ATTRS:
                    violations.append(f"forbidden attribute: {attr}")

    return violations


# ─── Restricted Globals ──────────────────────────────────────────────────────


def _build_restricted_globals() -> dict[str, Any]:
    """Build a minimal globals dict for exec()."""
    safe_builtins = {
        # types
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "frozenset": frozenset,
        "type": type,
        # functions
        "len": len,
        "min": min,
        "max": max,
        "abs": abs,
        "sum": sum,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "chr": chr,
        "ord": ord,
        # constants
        "True": True,
        "False": False,
        "None": None,
        # exceptions (needed for try/except)
        "ValueError": ValueError,
        "TypeError": TypeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "AttributeError": AttributeError,
        "StopIteration": StopIteration,
        "Exception": Exception,
    }

    # Allow `import re` — the only whitelisted import.
    # Without __import__ in builtins, even `import re` fails at runtime.
    _real_import = builtins.__import__

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in _ALLOWED_IMPORTS:
            return _real_import(name, *args, **kwargs)
        raise ImportError(f"import of '{name}' is not allowed")

    safe_builtins["__import__"] = _safe_import

    return {"__builtins__": safe_builtins, "re": re}


# ─── Execution ───────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CodeExecResult:
    success: bool
    returned_region: str
    error: str | None
    exec_time_ms: float
    error_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "returned_region_size": len(self.returned_region),
            "error": self.error,
            "exec_time_ms": self.exec_time_ms,
            "error_kind": self.error_kind,
        }


def _run_locate_region_in_subprocess(
    code: str,
    document_text: str,
    child_conn: Any,
) -> None:
    try:
        namespace = _build_restricted_globals()
        exec(code, namespace)  # noqa: S102 - intentional sandboxed exec
        fn = namespace.get("locate_region")
        if fn is None:
            child_conn.send(
                {
                    "success": False,
                    "error": "locate_region not defined in code",
                    "returned_region": "",
                }
            )
            return
        child_conn.send(
            {
                "success": True,
                "error": None,
                "returned_region": fn(document_text),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        child_conn.send(
            {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "returned_region": "",
            }
        )
    finally:
        child_conn.close()


def execute_locate_region(
    code: str,
    document_text: str,
    timeout_s: int = 5,
) -> CodeExecResult:
    """Execute a ``locate_region`` function inside a restricted sandbox.

    Steps:
      1. AST validation — reject unsafe code.
      2. ``exec()`` with restricted globals.
      3. Call ``locate_region(document_text)``.
      4. Validate the return is a non-empty substring.
      5. On any failure → no returned region.
    """
    t0 = time.perf_counter()

    # 1. AST check
    violations = validate_code_ast(code)
    if violations:
        dt = (time.perf_counter() - t0) * 1000
        return CodeExecResult(
            success=False,
            returned_region="",
            error=f"AST violations: {violations}",
            exec_time_ms=dt,
            error_kind="ast",
        )

    # 2-3. Run exec + call in a separate process for cross-platform, thread-safe timeout control.
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_run_locate_region_in_subprocess,
        args=(code, document_text, child_conn),
    )
    child_conn_open = True
    result_payload: dict[str, Any] | None = None
    try:
        process.start()
        child_conn.close()
        child_conn_open = False
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            remaining_s = deadline - time.perf_counter()
            if parent_conn.poll(min(0.05, remaining_s)):
                result_payload = parent_conn.recv()
                break
            if not process.is_alive():
                break

        if result_payload is None and process.is_alive():
            process.terminate()
            process.join()
            dt = (time.perf_counter() - t0) * 1000
            return CodeExecResult(
                success=False,
                returned_region="",
                error=f"TimeoutError: execution exceeded {timeout_s}s",
                exec_time_ms=dt,
                error_kind="timeout",
            )

        process.join()
        if result_payload is None:
            if parent_conn.poll():
                result_payload = parent_conn.recv()
            else:
                result_payload = {
                    "success": False,
                    "error": f"sandbox process exited with code {process.exitcode}",
                    "returned_region": "",
                }
    except Exception as exc:
        dt = (time.perf_counter() - t0) * 1000
        return CodeExecResult(
            success=False,
            returned_region="",
            error=f"{type(exc).__name__}: {exc}",
            exec_time_ms=dt,
            error_kind="runtime_error",
        )
    finally:
        if child_conn_open:
            child_conn.close()
        parent_conn.close()

    if not result_payload["success"]:
        dt = (time.perf_counter() - t0) * 1000
        return CodeExecResult(
            success=False,
            returned_region="",
            error=result_payload["error"],
            exec_time_ms=dt,
            error_kind="runtime_error",
        )

    result = result_payload["returned_region"]
    dt = (time.perf_counter() - t0) * 1000

    # 4. validate return value
    if not isinstance(result, str) or not result:
        return CodeExecResult(
            success=False,
            returned_region="",
            error=f"locate_region returned {type(result).__name__}, expected non-empty str",
            exec_time_ms=dt,
            error_kind="runtime_error",
        )

    # Verify the result is a substring of the document (allow minor whitespace differences).
    if result not in document_text:
        # Lenient fallback: retry after stripping surrounding whitespace.
        stripped = result.strip()
        if stripped and stripped in document_text:
            result = stripped
        else:
            return CodeExecResult(
                success=False,
                returned_region="",
                error="returned string is not a substring of document_text",
                exec_time_ms=dt,
                error_kind="runtime_error",
            )

    return CodeExecResult(
        success=True,
        returned_region=result,
        error=None,
        exec_time_ms=dt,
    )


__all__ = [
    "CodeExecResult",
    "execute_locate_region",
    "validate_code_ast",
]
