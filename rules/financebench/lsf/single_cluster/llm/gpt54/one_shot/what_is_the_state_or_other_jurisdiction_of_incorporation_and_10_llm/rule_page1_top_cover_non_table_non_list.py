def rule_page1_top_cover_non_table_non_list(doc: dict) -> list[dict]:
    """Match non-table, non-list page-1 spans in the top cover area that often contain the answer."""
    try:
        out = []
        count = 0
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                count += 1
                if count <= 35 and span.get("label") in ("text", "section_header"):
                    out.append(span)
        return out
    except Exception:
        return []
