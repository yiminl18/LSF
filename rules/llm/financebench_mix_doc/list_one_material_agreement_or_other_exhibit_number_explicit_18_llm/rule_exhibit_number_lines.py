def rule_exhibit_number_lines(doc: dict) -> list[dict]:
    """Match spans containing exhibit-number patterns like 'Exhibit 10.1' or '10.1' with exhibit context."""
    import re
    try:
        out = []
        pat1 = re.compile(r"\bexhibit\s+\d+(?:\.\d+)?[a-z]?\b", re.I)
        pat2 = re.compile(r"\b\d+(?:\.\d+)?[a-z]?\b")
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if pat1.search(txt):
                out.append(span)
            elif "exhibit" in txt.lower() and pat2.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
