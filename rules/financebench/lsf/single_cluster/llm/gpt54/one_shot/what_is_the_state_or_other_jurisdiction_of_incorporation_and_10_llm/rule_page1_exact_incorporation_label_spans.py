def rule_page1_exact_incorporation_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans whose text is the incorporation-jurisdiction label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(
                r"^\(?\s*(state|state or other jurisdiction)( of)? incorporation( or organization)?\s*\)?$|^\(?\s*state or other jurisdiction of incorporation or organization\s*\)?$|^\(?\s*state of incorporation\s*\)?$",
                text,
                re.I,
            ):
                out.append(span)
        return out
    except Exception:
        return []
