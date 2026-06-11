def rule_page1_numeric_ein_pattern(doc: dict) -> list[dict]:
    """Match page-1 spans that look like EIN values (NN-NNNNNNN)."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
        return out
    except Exception:
        return []
