def rule_page1_h1_allcaps_large(doc: dict) -> list[dict]:
    """Match large bold page-1 H1 headers in all caps that are not SEC/form boilerplate."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and span.get("bold") == 1
                and float(span.get("size") or 0) >= 10
                and txt
                and "form 10-" not in low
                and "securities and exchange commission" not in low
                and "united states" not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
