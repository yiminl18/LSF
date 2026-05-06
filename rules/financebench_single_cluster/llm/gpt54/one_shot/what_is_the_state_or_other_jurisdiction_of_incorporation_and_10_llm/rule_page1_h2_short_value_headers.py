def rule_page1_h2_short_value_headers(doc: dict) -> list[dict]:
    """Match short H2 page-1 section headers that often hold the state or EIN value."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            level = span.get("structure", {}).get("level")
            if span.get("page_no") == 1 and span.get("label") == "section_header" and level == "H2":
                if len(text.split()) <= 6 and (re.fullmatch(r"\d{2}-\d{7}", text) or not re.search(r"commission|form 10-k|part i|table of contents", text, re.I)):
                    out.append(span)
        return out
    except Exception:
        return []
