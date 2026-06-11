def rule_page1_last_h1(doc: dict) -> list[dict]:
    """Match the last H1 section_header on page 1."""
    try:
        texts = doc.get("texts", [])
        cands = [
            s for s in texts
            if s.get("page_no") == 1
            and s.get("label") == "section_header"
            and s.get("structure", {}).get("level") == "H1"
        ]
        return cands[-1:] if cands else []
    except Exception:
        return []
