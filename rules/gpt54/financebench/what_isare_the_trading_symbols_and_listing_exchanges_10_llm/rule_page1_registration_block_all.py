def rule_page1_registration_block_all(doc: dict) -> list[dict]:
    """Return all page 1 spans from the first 12(b) registration mention until the 12(g) mention or end of top block."""
    try:
        import re
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if start is None and span.get("page_no") == 1 and re.search(r"section\s+12\(b\)|12\(b\)\s+of\s+the\s+act", txt, re.I):
                start = i
            if start is not None and span.get("page_no") == 1 and re.search(r"section\s+12\(g\)|12\(g\)\s+of\s+the\s+act", txt, re.I):
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 20)
        return [s for s in texts[start:end] if s.get("page_no") == 1]
    except Exception:
        return []
