def rule_page1_body_text_address_line(doc: dict) -> list[dict]:
    """Match page-1 body text spans that contain the full principal office address line."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "text":
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'\d{1,6}\s+\S+', text) and (
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}', text) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+\d{4,})?', text) or
                re.search(r'\bUnited Kingdom\b', text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
