def rule_8k_exhibit_line_items(doc: dict) -> list[dict]:
    """Match 8-K body text lines that start with exhibit numbers like 3.2, 4.6, 99.1, or 104."""
    import re
    try:
        out = []
        pat = re.compile(r"^\s*(?:exhibit\s*)?(?:3\.2|4\.6|4\.7|10\.1|10\.24|99\.1|104)\b", re.I)
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
