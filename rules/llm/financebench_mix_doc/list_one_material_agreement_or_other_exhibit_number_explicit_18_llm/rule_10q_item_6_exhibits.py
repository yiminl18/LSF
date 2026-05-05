def rule_10q_item_6_exhibits(doc: dict) -> list[dict]:
    """Match 10-Q TOC or headers for Item 6 Exhibits."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if re.search(r"\bitem\s*6\b", txt, re.I) and re.search(r"\bexhibits?\b", txt, re.I):
                out.append(span)
            elif "item 6" in path.lower() and "exhibit" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
