def rule_page1_h3_short_identifier_headers(doc: dict) -> list[dict]:
    """Match short H3 page-1 headers that are often OCR-split EIN or nearby cover-page identifiers."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H3"
                and len(text) <= 60
                and (
                    re.fullmatch(r"\d{2}-\d{7}", text)
                    or re.search(r"i\.?r\.?s\.?|state|jurisdiction", text, re.I)
                )
            ):
                out.append(span)
        return out
    except Exception:
        return []
