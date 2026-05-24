def rule_page1_cover_page_first_25_spans_telephone_label(doc: dict) -> list[dict]:
    """Match telephone-label spans among the first 25 spans on the cover page."""
    try:
        import re
        out = []
        for span in (doc.get("texts", []) or [])[:25]:
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
