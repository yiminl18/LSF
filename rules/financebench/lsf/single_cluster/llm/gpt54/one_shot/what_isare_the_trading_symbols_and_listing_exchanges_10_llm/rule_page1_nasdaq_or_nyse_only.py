def rule_page1_nasdaq_or_nyse_only(doc: dict) -> list[dict]:
    """Match any page 1 span containing only the core exchange names or abbreviations."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.search(r"\bnyse\b", txt, re.I):
                out.append(span)
            elif re.search(r"nasdaq", txt, re.I):
                out.append(span)
            elif re.search(r"new york stock exchange", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
