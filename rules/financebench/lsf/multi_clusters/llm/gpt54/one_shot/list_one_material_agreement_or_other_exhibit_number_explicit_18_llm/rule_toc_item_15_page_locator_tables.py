def rule_toc_item_15_page_locator_tables(doc: dict) -> list[dict]:
    """Match TOC tables that contain Item 15 Exhibits rows, useful as page locators for the real Exhibit Index."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r"\bitem\s*15\b", txt, re.I) and re.search(r"\bexhibits?\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
