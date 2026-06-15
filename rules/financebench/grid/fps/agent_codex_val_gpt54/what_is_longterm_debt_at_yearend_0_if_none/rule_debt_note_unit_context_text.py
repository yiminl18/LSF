def rule_debt_note_unit_context_text(doc: dict) -> list[dict]:
    """Match unit-bearing text immediately before debt-note tables."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            path = (((span.get("structure") or {}).get("path_text") or "")).lower()
            text = (span.get("text") or "").lower()
            if (
                span.get("label") == "table"
                and "debt" in path
                and any(
                    key in text
                    for key in (
                        "total long-term debt",
                        "long-term debt, excluding current portion",
                        "long-term debt, less current portion",
                        "long-term debt, net",
                        "carrying value of long-term debt",
                        "total non-current portion of term debt",
                        "total term debt",
                        "adjusted carrying value of long-term debt",
                    )
                )
                and i > 0
            ):
                prev = texts[i - 1]
                prev_text = (prev.get("text") or "").lower()
                prev_path = (((prev.get("structure") or {}).get("path_text") or "")).lower()
                if (
                    prev.get("label") == "text"
                    and ("debt" in prev_path or "debt" in path)
                    and any(
                        cue in prev_text
                        for cue in ("in thousands", "in millions", "in billions", "consisted of the following")
                    )
                ):
                    out.append(prev)
        return out
    except Exception:
        return []
