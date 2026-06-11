def rule_page1_h2_state_or_ein_headers(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 section headers whose title is the state or EIN value."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            lvl = span.get("structure", {}).get("level")
            if span.get("page_no") == 1 and span.get("label") == "section_header" and lvl in {"H2", "H3"}:
                if re.fullmatch(r"\d{2}-\d{7}", text) or re.fullmatch(r"[A-Z][A-Za-z]+(?:\s*\([A-Za-z ]+\))?", text):
                    out.append(span)
        return out
    except Exception:
        return []
