def rule_page1_short_numeric_or_state_h2s(doc: dict) -> list[dict]:
    """Match page-1 H2 headers that are either EIN-like numbers or short state/jurisdiction values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            if span.get("structure", {}).get("level") != "H2":
                continue
            text = (span.get("text") or "").strip()
            if re.fullmatch(r"\d{2}-\d{7}", text) or (1 <= len(text.split()) <= 4 and not re.search(r"\d", text)):
                out.append(span)
        return out
    except Exception:
        return []
