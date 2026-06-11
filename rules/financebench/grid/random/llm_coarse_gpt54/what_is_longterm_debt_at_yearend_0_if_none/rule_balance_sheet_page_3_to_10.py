def rule_balance_sheet_page_3_to_10(doc: dict) -> list[dict]:
    """Match likely balance sheet tables on common financial statement pages 3-10."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no")
            if page is None or not (3 <= page <= 10):
                continue
            text = span.get("text", "") or ""
            path = (span.get("structure") or {}).get("path_text", "") or ""
            if re.search(r"balance sheet", text + " " + path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
