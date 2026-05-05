def rule_tables_with_total_assets_within_three_pages_of_item8(doc: dict) -> list[dict]:
    """Match tables within three pages after Item 8 heading that contain total assets or assets."""
    try:
        item8_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if "item 8" in txt or "financial statements and supplementary data" in txt:
                    item8_pages.add(int(span.get("page_no", -999)))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            p = int(span.get("page_no", -999))
            if any(base <= p <= base + 3 for base in item8_pages):
                txt = (span.get("text") or "").lower()
                if "assets" in txt or "balance sheet" in txt or "financial position" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
