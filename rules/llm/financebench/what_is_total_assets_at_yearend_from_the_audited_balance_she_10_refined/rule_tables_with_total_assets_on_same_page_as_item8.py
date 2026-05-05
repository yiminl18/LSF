def rule_tables_with_total_assets_on_same_page_as_item8(doc: dict) -> list[dict]:
    """Match tables on pages containing an Item 8 heading and total assets."""
    try:
        item8_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if "item 8" in txt or "financial statements and supplementary data" in txt:
                    item8_pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in item8_pages:
                if "total assets" in (span.get("text") or "").lower() or "assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
