"""Minimal LangChain tool-calling agent on Azure OpenAI (same config as :mod:`models.gpt54`).

Streaming / progress: LangChain’s `stream_mode` API for :func:`create_agent` is described at
https://docs.langchain.com/oss/python/langchain/streaming — this project uses LangChain 0.3
:class:`~langchain.agents.AgentExecutor`, so :func:`run_agent` uses ``astream_events``
instead to surface tool/LLM steps and token usage.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import datetime
import json
import logging

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

# langchain_core.language_models.base imports BaseCache, Callbacks, and LLMResult
# only under TYPE_CHECKING, so pydantic v2 cannot resolve them at runtime.
# Inject into every relevant module's globals and pass _types_namespace to model_rebuild.
try:
    import langchain_core.language_models.base as _lm_base
    import langchain_openai.chat_models.azure as _azure_mod
    import langchain_openai.chat_models.base as _openai_base
    from langchain_core.caches import BaseCache as _BaseCache
    from langchain_core.callbacks import Callbacks as _Callbacks
    from langchain_core.outputs import LLMResult as _LLMResult
    _ns = {"BaseCache": _BaseCache, "Callbacks": _Callbacks, "LLMResult": _LLMResult}
    for _mod in (_lm_base, _azure_mod, _openai_base):
        _mod.BaseCache = _BaseCache      # type: ignore[attr-defined]
        _mod.Callbacks = _Callbacks      # type: ignore[attr-defined]
        _mod.LLMResult = _LLMResult      # type: ignore[attr-defined]
    AzureChatOpenAI.model_rebuild(_types_namespace=_ns)
except Exception:
    pass

from models import gpt54 as _gpt
from tools.grab import grab
from tools.search import search
from tools.read_json import read_json
from tools.read_pdf_chart import read_pdf_chart
from tools.grabtoc import grabtoc
from tools.process_page_image import process_page_image


@tool
def grep(text: str = "", keyword: str = "", path: str = "") -> str:
    """Search paragraphs for ``keyword``. Use ``path`` to load a UTF-8 text file, or pass raw ``text``."""
    from pathlib import Path as P

    from tools.grab import grab_passages

    p = (path or "").strip()
    body = P(p).read_text(encoding="utf-8", errors="replace") if p else text
    kw = (keyword or "").strip()
    if not kw:
        return "(no keyword)"
    hits = grab_passages(body, kw)
    return "\n---\n".join(hits) if hits else "(no matches)"


_REGISTRY: dict[str, Any] = {
    "grep": grep,
    "grab": grab,
    "search": search,
    "read_json": read_json,
    "read_pdf_chart": read_pdf_chart,
    "grabtoc": grabtoc,
    "process_page_image": process_page_image,
}


def register_tool(name: str, fn: Any) -> None:
    """Bind ``name`` to a LangChain tool or plain callable (wrapped with :func:`tool`)."""
    _REGISTRY[name] = fn if hasattr(fn, "invoke") else tool(fn)


def _azure_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=_gpt.AZURE_ENDPOINT,
        api_key=_gpt.api_key,
        api_version=_gpt.AZURE_API_VERSION,
        azure_deployment=_gpt.AZURE_DEPLOYMENT,
        temperature=0.0,
        model_kwargs={"stream_options": {"include_usage": True}},
    )


def create_agent(
    tool_names: list[str],
    *,
    system_prompt: str = "You are a helpful assistant. Use tools when they help answer the user.",
    verbose: bool = False,
) -> AgentExecutor:
    tools = []
    for n in tool_names:
        if n not in _REGISTRY:
            raise ValueError(f"unknown tool {n!r}; registered: {sorted(_REGISTRY)}")
        tools.append(_REGISTRY[n])
    llm = _azure_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=30,
        handle_parsing_errors=True,
    )


def _usage_delta(msg: Any) -> dict[str, int]:
    """Collect token counts from ``AIMessage`` (LangChain OpenAI uses ``usage_metadata``)."""
    out: dict[str, int] = {}

    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        try:
            if um.get("input_tokens") is not None:
                out["prompt_tokens"] = int(um["input_tokens"])
            if um.get("output_tokens") is not None:
                out["completion_tokens"] = int(um["output_tokens"])
            if um.get("total_tokens") is not None:
                out["total_tokens"] = int(um["total_tokens"])
        except (TypeError, ValueError):
            pass
    elif um is not None:
        for attr, key in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            v = getattr(um, attr, None)
            if v is not None:
                try:
                    out[key] = int(v)
                except (TypeError, ValueError):
                    pass

    meta = getattr(msg, "response_metadata", None) or {}
    usage = meta.get("token_usage") if isinstance(meta, dict) else None
    if isinstance(usage, dict):
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if k in out:
                continue
            v = usage.get(k)
            if v is not None:
                try:
                    out[k] = int(v)
                except (TypeError, ValueError):
                    pass

    if out.get("total_tokens", 0) == 0 and (
        out.get("prompt_tokens", 0) or out.get("completion_tokens", 0)
    ):
        out["total_tokens"] = out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
    return out


def _handle_stream_event(
    event: dict[str, Any],
    state: dict[str, Any],
    *,
    on_progress: Callable[[str], None] | None,
    stream_tokens: bool,
) -> None:
    et = event.get("event")
    name = event.get("name", "")
    data = event.get("data") or {}

    def emit(msg: str) -> None:
        state["progress"].append(msg)
        if on_progress:
            on_progress(msg)

    if et == "on_tool_start":
        inp = data.get("input")
        emit(f"[progress] tool_start {name} input={str(inp)[:400]!r}")
        state["tool_calls"].append({"name": name, "input": inp, "output": None})
    elif et == "on_tool_end":
        raw = data.get("output", "")
        full_output = str(raw)
        if state["tool_calls"]:
            state["tool_calls"][-1]["output"] = full_output
        prev = full_output[:400].replace("\n", " ")
        emit(f"[progress] tool_end {name} output_preview={prev!r}")
    elif et == "on_chat_model_start":
        emit(f"[progress] llm_start {name}")
    elif et == "on_chat_model_stream" and stream_tokens:
        chunk = data.get("chunk")
        piece = getattr(chunk, "content", None)
        if piece:
            emit(f"[token] {piece!r}")
    elif et == "on_chat_model_end":
        delta = _usage_delta(data.get("output"))
        for k, v in delta.items():
            state["usage"][k] = state["usage"].get(k, 0) + v
        msg = data.get("output")
        if msg is not None and getattr(msg, "content", None):
            c = msg.content
            if isinstance(c, str) and c.strip():
                state["last_ai"] = c
        emit(
            "[progress] llm_end "
            f"step p={delta.get('prompt_tokens', 0)} c={delta.get('completion_tokens', 0)} "
            f"t={delta.get('total_tokens', 0)} | "
            f"cumul p={state['usage'].get('prompt_tokens', 0)} "
            f"c={state['usage'].get('completion_tokens', 0)}"
        )
    elif et == "on_chain_end":
        out = data.get("output")
        if isinstance(out, dict) and "output" in out:
            state["final_output"] = out.get("output")


async def _arun_streaming(
    executor: AgentExecutor,
    user_message: str,
    *,
    on_progress: Callable[[str], None] | None,
    stream_tokens: bool,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "progress": [],
        "usage": {},
        "final_output": None,
        "last_ai": None,
        "tool_calls": [],
    }
    async for event in executor.astream_events({"input": user_message}, version="v2"):
        _handle_stream_event(event, state, on_progress=on_progress, stream_tokens=stream_tokens)

    pt = int(state["usage"].get("prompt_tokens", 0) or 0)
    ct = int(state["usage"].get("completion_tokens", 0) or 0)
    tt_reported = int(state["usage"].get("total_tokens", 0) or 0)
    total = tt_reported if tt_reported >= (pt + ct) else pt + ct

    out_text = state["final_output"]
    if out_text is None:
        out_text = state["last_ai"]
    if out_text is None:
        out_text = ""

    return {
        "input": user_message,
        "output": out_text,
        "token_usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": total,
        },
        "progress": state["progress"],
        "tool_calls": state["tool_calls"],
    }


def run_agent(
    executor: AgentExecutor,
    user_message: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    stream: bool = True,
    stream_tokens: bool = False,
) -> dict[str, Any]:
    """
    Execute the agent.

    * ``stream=True`` (default): async :meth:`~Runnable.astream_events` — fills ``progress``
      (step lines) and ``token_usage`` (sums Azure/OpenAI usage from each LLM call).
    * ``stream_tokens=True``: also emit one progress line per streamed text chunk (verbose).
    * ``stream=False``: single :meth:`~Runnable.invoke` (no ``progress`` / ``token_usage``).
    """
    if stream:
        return asyncio.run(
            _arun_streaming(
                executor,
                user_message,
                on_progress=on_progress,
                stream_tokens=stream_tokens,
            )
        )
    return executor.invoke({"input": user_message})


async def arun_agent(
    executor: AgentExecutor,
    user_message: str,
    *,
    on_progress: Callable[[str], None] | None = None,
    stream_tokens: bool = False,
) -> dict[str, Any]:
    """Async variant of :func:`run_agent` with ``stream=True``."""
    return await _arun_streaming(
        executor,
        user_message,
        on_progress=on_progress,
        stream_tokens=stream_tokens,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def _write_log(log_dir: Path, data: dict[str, Any]) -> Path:
    """Write a JSON run log to ``log_dir`` with a timestamped filename."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"run_{ts}.json"
    log_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_file


def run_on_data(
    data_link: str,
    query: str,
    tools: list[str],
    *,
    log_dir: str = "logs/officeqa",
    text_subdir: str = "text",
    system_prompt: str | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """Run the agent over documents in ``data_link`` using the named tools.

    Args:
        data_link:    Path to the data directory (must contain a ``text/`` sub-folder
                      with ``.txt`` files, unless ``text_subdir`` is overridden).
        query:        The user question to answer.
        tools:        List of tool names to make available (e.g. ``["grab", "search"]``).
        log_dir:      Directory for JSON run logs.
        text_subdir:  Sub-folder inside ``data_link`` containing ``.txt`` files.
        system_prompt: Override the default system prompt.
        stream:       Whether to use streaming (captures progress + token usage).

    Returns:
        dict with keys ``input``, ``output``, ``token_usage``, ``progress``,
        ``log_file`` (path of the written log).
    """
    data_path = Path(data_link)
    text_dir = data_path / text_subdir
    json_dir = data_path / "json"

    # Discover available text and JSON files
    if text_dir.is_dir():
        txt_files = sorted(text_dir.glob("*.txt"))
    else:
        txt_files = sorted(data_path.glob("*.txt"))

    json_files = sorted(json_dir.glob("*.json")) if json_dir.is_dir() else []

    def _flist(files: list[Path]) -> str:
        return "\n".join(f"  - {f.resolve()}" for f in files)

    file_note_parts = []
    if txt_files:
        file_note_parts.append(f"Text files (use with grab/search tools):\n{_flist(txt_files)}")
    if json_files:
        file_note_parts.append(
            f"Structured JSON files (use with read_json tool — contain tables, "
            f"figures, and section headers extracted from the PDFs):\n{_flist(json_files)}"
        )
    file_note = "\n\n".join(file_note_parts) if file_note_parts else f"No files found under {data_path}."

    if system_prompt is None:
        system_prompt = (
            "You are a precise document analyst with access to document retrieval tools.\n\n"
            f"{file_note}\n\n"
            "IMPORTANT NOTES:\n"
            "- Chart and figure data in the PDFs may NOT be present in the text files "
            "(figures have no extracted content). However, narrative text often mentions "
            "key values in context.\n"
            "- The read_json tool lets you access structured JSON documents that may "
            "contain table data not easily found in the text files.\n"
            "- Use grab to find paragraphs containing specific keywords.\n"
            "- Use search for semantic similarity search over document text.\n"
            "- When numerical computation is required, show each step explicitly.\n"
            "- If specific values are not directly in the text, look in surrounding "
            "narrative paragraphs for mentions of the same data points."
        )

    executor = create_agent(tools, system_prompt=system_prompt)

    log_data: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "data_link": str(data_link),
        "query": query,
        "tools": tools,
        "steps": [],
        "output": None,
        "token_usage": {},
    }

    progress_lines: list[str] = []

    def _on_progress(msg: str) -> None:
        progress_lines.append(msg)
        log_data["steps"].append(msg)

    result = run_agent(
        executor,
        query,
        on_progress=_on_progress,
        stream=stream,
    )

    log_data["output"] = result.get("output")
    log_data["token_usage"] = result.get("token_usage", {})
    log_data["progress"] = progress_lines

    log_path = _write_log(Path(log_dir), log_data)
    result["log_file"] = str(log_path)
    return result
