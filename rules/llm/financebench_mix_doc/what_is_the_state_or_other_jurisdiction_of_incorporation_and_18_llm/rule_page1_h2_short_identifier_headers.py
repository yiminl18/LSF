def rule_page1_h2_short_identifier_headers(doc: dict) -> list[dict]:
    """Match short H2 page-1 headers that are often OCR-split state or EIN values."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text", "") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H2"
                and len(text) <= 50
                and (
                    re.fullmatch(r"\d{2}-\d{7}", text)
                    or re.fullmatch(r"[A-Za-z][A-Za-z .,&()\-]{1,40}", text)
                )
            ):
                out.append(span)
        return out
    except Exception:
        return []
