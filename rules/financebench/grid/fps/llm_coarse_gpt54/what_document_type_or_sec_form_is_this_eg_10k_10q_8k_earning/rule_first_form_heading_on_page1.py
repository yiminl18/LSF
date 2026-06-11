def rule_first_form_heading_on_page1(doc: dict) -> list[dict]:
    """Return the earliest page-1 span that looks like a FORM heading."""
    import re
    try:
        candidates = []
        for i, span in enumerate(doc.get("texts", [])):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.match(r"^FORM\s+[A-Z0-9\-]+", text, re.I):
                candidates.append((i, span))
        return [candidates[0][1]] if candidates else []
    except Exception:
        return []
