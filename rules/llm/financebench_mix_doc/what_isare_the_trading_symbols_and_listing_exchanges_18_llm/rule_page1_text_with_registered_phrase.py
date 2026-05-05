def rule_page1_text_with_registered_phrase(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'registered' in listing context."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "registered pursuant to section 12(b)" in txt
                or "on which registered" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
