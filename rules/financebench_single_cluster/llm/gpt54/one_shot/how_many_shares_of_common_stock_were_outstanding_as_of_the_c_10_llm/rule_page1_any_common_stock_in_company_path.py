def rule_page1_any_common_stock_in_company_path(doc: dict) -> list[dict]:
    """Match any page-1/2 span under a non-Item company path that contains 'common stock'."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            t = (span.get("text") or "").lower()
            if span.get("page_no") in (1, 2) and "common stock" in t:
                if "item " not in path and "risk factors" not in path and "table of contents" not in path and "index" not in path:
                    out.append(span)
        return out
    except Exception:
        return []
