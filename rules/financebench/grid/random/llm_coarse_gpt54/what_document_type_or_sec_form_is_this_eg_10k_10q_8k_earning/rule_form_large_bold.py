def rule_form_large_bold(doc: dict) -> list[dict]:
    """Match large bold spans containing a common SEC form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I):
                    out.append(span)
        return out
    except Exception:
        return []
