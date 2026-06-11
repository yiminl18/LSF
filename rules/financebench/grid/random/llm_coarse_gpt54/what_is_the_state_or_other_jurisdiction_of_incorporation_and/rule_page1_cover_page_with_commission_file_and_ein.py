def rule_page1_cover_page_with_commission_file_and_ein(doc: dict) -> list[dict]:
    """Match page-1 spans containing commission file wording and EIN pattern, common in 8-K cover blocks."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"Commission File", txt, re.I) and re.search(r"\b\d{2}-\d{7}\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
