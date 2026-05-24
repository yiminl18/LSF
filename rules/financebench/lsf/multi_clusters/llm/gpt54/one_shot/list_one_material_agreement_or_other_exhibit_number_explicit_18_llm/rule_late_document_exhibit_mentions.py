def rule_late_document_exhibit_mentions(doc: dict) -> list[dict]:
    """Match exhibit-related spans appearing in the last quarter of the document, where Exhibit Index usually lives."""
    import re
    try:
        texts = doc.get("texts", [])
        if not texts:
            return []
        start = int(len(texts) * 0.75)
        out = []
        for span in texts[start:]:
            txt = (span.get("text") or "")
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if re.search(r"\bexhibit\b", txt, re.I) or "exhibit" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
