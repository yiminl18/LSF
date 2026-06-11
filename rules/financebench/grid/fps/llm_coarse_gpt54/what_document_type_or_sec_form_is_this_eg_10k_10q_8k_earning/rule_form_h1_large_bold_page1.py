def rule_form_h1_large_bold_page1(doc: dict) -> list[dict]:
    """Match large bold H1 page-1 section headers that look like FORM headings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("bold") == 1
                and float(span.get("size") or 0) >= 10
                and str(span.get("structure", {}).get("level", "")) == "H1"
                and re.search(r"\bFORM\b", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
