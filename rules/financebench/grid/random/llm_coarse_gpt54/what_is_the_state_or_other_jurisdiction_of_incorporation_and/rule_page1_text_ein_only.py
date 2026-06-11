def rule_page1_text_ein_only(doc: dict) -> list[dict]:
    """Match page-1 spans that are just an EIN-like number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r"\d{2}-\d{7}", text):
                out.append(span)
        return out
    except Exception:
        return []
