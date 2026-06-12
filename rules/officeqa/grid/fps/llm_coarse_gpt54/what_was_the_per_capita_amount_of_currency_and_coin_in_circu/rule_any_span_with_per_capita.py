def rule_any_span_with_per_capita(doc: dict) -> list[dict]:
    """Match any span mentioning per capita, a strong cue for the answer-bearing table."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if re.search(r'per\s+capita', (span.get("text") or ""), re.I):
                out.append(span)
    except Exception:
        return []
    return out
