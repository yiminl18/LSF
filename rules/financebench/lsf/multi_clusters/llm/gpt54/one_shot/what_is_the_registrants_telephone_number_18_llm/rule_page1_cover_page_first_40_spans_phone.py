def rule_page1_cover_page_first_40_spans_phone(doc: dict) -> list[dict]:
    """Match phone-bearing spans among the first 40 spans, capturing cover-page metadata blocks."""
    try:
        import re
        out = []
        for span in (doc.get("texts", []) or [])[:40]:
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
