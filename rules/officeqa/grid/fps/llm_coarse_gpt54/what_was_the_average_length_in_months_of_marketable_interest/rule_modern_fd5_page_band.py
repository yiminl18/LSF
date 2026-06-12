def rule_modern_fd5_page_band(doc: dict) -> list[dict]:
    """Match spans on pages where modern contents place FD-5 (typically around pages 15-36 depending on era) and mention average length."""
    out = []
    try:
        for span in doc.get("texts", []):
            p = span.get("page_no")
            txt = (span.get("text") or "").lower()
            if p is not None and 15 <= p <= 40 and "average length" in txt and "marketable" in txt:
                out.append(span)
    except Exception:
        return []
    return out
