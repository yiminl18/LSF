def rule_item1_business_first_paragraph_with_offices(doc: dict) -> list[dict]:
    """Match early Item 1/Business paragraphs mentioning offices location."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            idx = ((span.get("structure") or {}).get("sibling_index_norm") or 0)
            txt = (span.get("text") or "").strip()
            if re.search(r'Item 1|Business', path, re.I) and idx <= 0.2 and re.search(r'offices.*located|facilities.*located', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
