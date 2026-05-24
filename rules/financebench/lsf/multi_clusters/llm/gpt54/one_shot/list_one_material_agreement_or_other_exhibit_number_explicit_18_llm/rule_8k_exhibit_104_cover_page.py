def rule_8k_exhibit_104_cover_page(doc: dict) -> list[dict]:
    """Match spans mentioning Exhibit 104 / Cover Page Interactive Data File."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"\b104\b", txt) and re.search(r"cover page interactive data file", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
