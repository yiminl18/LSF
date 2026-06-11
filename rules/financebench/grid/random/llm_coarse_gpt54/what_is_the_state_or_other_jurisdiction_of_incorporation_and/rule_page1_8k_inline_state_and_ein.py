def rule_page1_8k_inline_state_and_ein(doc: dict) -> list[dict]:
    """Match page-1 8-K spans containing both a state/jurisdiction value and an EIN pattern."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"\b\d{2}-\d{7}\b", txt):
                if re.search(r"\bWashington\b|\bNew York\b|\bDelaware\b|\bJersey\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
