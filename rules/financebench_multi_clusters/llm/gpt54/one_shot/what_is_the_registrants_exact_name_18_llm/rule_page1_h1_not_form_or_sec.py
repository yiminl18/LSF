def rule_page1_h1_not_form_or_sec(doc: dict) -> list[dict]:
    """Match all page-1 H1 section headers excluding SEC/form boilerplate."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip().lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "form 10-" not in txt
                and "form 8-k" not in txt
                and "current report" not in txt
                and "securities and exchange commission" not in txt
                and "washington, d.c." not in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
