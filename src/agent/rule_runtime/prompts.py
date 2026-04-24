"""Prompt template registry and rendering helpers for rule-generation agents."""

from __future__ import annotations

from pathlib import Path

from agent.rule_runtime.data import DocumentSample, QueryPackage

# Version -> filename map
_PROMPT_FILENAMES: dict[str, str] = {
    "range_v1": "reflection_agent/range_rule_generation_v1.txt",
    "range_reflect_v1": "reflection_agent/range_rule_reflection_v1.txt",
    "code_scope_v1": "reflection_agent/code_scope_narrowing_v1.txt",
    "tool_agent_system_v2": "tool_agent/tool_agent_system_v2.txt",
    "tool_agent_system_diverse": "tool_agent/tool_agent_system_diverse.txt",
    "tool_agent_system_curriculum": "tool_agent/tool_agent_system_curriculum.txt",
}

# Loop markers for doc loop and answer loop (physically separated to avoid LLM confusion)
_DOC_LOOP_START = "{for each doc in documents:}"
_DOC_LOOP_END = "{end for documents}"
_ANS_LOOP_START = "{for each doc in answers:}"
_ANS_LOOP_END = "{end for answers}"


def load_prompt_template(version: str) -> str:
    """
    Load a prompt template file.

    version: a key in _PROMPT_FILENAMES.
    Path: src/agent/prompts/<category>/*.txt
    """
    filename = _PROMPT_FILENAMES.get(version)
    if filename is None:
        raise ValueError(
            f"Unknown prompt version: {version}; supported: {list(_PROMPT_FILENAMES)}"
        )

    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    template_path = prompts_dir / filename
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def _expand_loop(
    template: str,
    start_marker: str,
    end_marker: str,
    documents: list[DocumentSample],
    anonymize: bool,
    anon_labels: list[str],
) -> str:
    """Expand one loop region in the template, substituting {doc_id}, {markdown_text}, {ground_truth_answer}."""
    start = template.find(start_marker)
    end = template.find(end_marker)
    if start == -1 or end == -1:
        raise ValueError(f"Template missing loop marker '{start_marker}' or '{end_marker}'")

    fragment = template[start + len(start_marker) : end]
    blocks: list[str] = []
    for i, doc in enumerate(documents):
        block = fragment
        display_id = f"Document {anon_labels[i]}" if anonymize else doc.doc_id
        block = block.replace("{doc_id}", display_id)
        block = block.replace("{markdown_text}", doc.markdown_text)
        block = block.replace("{ground_truth_answer}", doc.ground_truth_answer)
        blocks.append(block)

    region = template[start : end + len(end_marker)]
    return template.replace(region, "".join(blocks))


def fill_prompt(
    template: str, query_package: QueryPackage, anonymize: bool = False
) -> str:
    """
    Fill a prompt template from a QueryPackage.

    Template placeholders:
    - {query_text} -> query_package.query_text
    - {for each doc in documents:} ... {end for documents} -> document content loop
    - {for each doc in answers:} ... {end for answers} -> answer loop (physically separate)

    anonymize: when True, hides real filenames and substitutes Document A/B/C/...
    """
    anon_labels = [chr(ord("A") + i) for i in range(26)]
    docs = query_package.documents

    # Expand document loop
    result = _expand_loop(
        template, _DOC_LOOP_START, _DOC_LOOP_END, docs, anonymize, anon_labels
    )

    # Expand answer loop if present
    if _ANS_LOOP_START in result:
        result = _expand_loop(
            result, _ANS_LOOP_START, _ANS_LOOP_END, docs, anonymize, anon_labels
        )

    result = result.replace("{query_text}", query_package.query_text)
    result = result.replace("{query_idx}", str(query_package.query_idx))
    return result
