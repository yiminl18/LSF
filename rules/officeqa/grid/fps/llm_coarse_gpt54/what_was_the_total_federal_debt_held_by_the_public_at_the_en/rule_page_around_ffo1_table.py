def rule_page_around_ffo1_table(doc: dict) -> list[dict]:
    """Match tables on pages where a nearby header/title says Table FFO-1 / Summary of Fiscal Operations."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if span.get("label") in {"section_header", "text"} and (
                "table ffo-1" in txt or "table ff0-1" in txt or "summary of fiscal operations" in txt
            ):
                pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in pages:
                out.append(span)
        return out
    except Exception:
        return []
