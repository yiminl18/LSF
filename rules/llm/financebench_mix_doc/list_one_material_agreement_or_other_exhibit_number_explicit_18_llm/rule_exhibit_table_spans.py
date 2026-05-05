def rule_exhibit_table_spans(doc: dict) -> list[dict]:
    """Match table spans whose markdown text contains exhibit-number patterns."""
    import re
    try:
        out = []
        pat = re.compile(r"\b(?:exhibit\s+)?\d+(?:\.\d+)?[a-z]?\b", re.I)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if "exhibit" in txt.lower() and pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
