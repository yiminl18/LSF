def rule_none_listed_signal_no_exhibit_10(doc: dict) -> list[dict]:
    """Return TOC/header exhibit spans in documents that appear to lack Exhibit 10.x references, useful for 'none listed' cases."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        has_ex10 = any(re.search(r"\bExhibit\s+10(?:\.\d+)?[A-Za-z]?\b", s.get("text") or "", re.I) for s in texts)
        if not has_ex10:
            for span in texts:
                txt = span.get("text") or ""
                path = ((span.get("structure") or {}).get("path_text") or "")
                if re.search(r"\b(exhibit index|exhibits?)\b", txt + " " + path, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
