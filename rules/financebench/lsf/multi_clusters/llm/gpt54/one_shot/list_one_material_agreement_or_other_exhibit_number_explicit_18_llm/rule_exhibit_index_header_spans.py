def rule_exhibit_index_header_spans(doc: dict) -> list[dict]:
    """Match section headers or text spans explicitly labeled 'Exhibit Index'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if re.search(r"\bexhibit index\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
