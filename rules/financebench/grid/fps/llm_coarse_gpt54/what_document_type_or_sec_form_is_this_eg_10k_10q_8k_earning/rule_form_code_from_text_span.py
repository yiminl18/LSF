def rule_form_code_from_text_span(doc: dict) -> list[dict]:
    """Match spans whose text_span contains a form code even if the text itself is noisy."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combo = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", combo, re.I):
                out.append(span)
        return out
    except Exception:
        return []
