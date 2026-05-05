def rule_item8_or_financial_statements_pages(doc: dict) -> list[dict]:
    """Match spans on pages whose path indicates Item 8 or Financial Statements, favoring tables and nearby headers."""
    try:
        pages = set()
        for span in doc.get("texts", []):
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            txt = (span.get("text") or "").lower()
            if "item 8" in path or "financial statements" in path or "supplementary data" in path or (
                span.get("label") == "section_header" and "statement of income" in txt
            ):
                if span.get("page_no") is not None:
                    pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") in pages and span.get("label") in {"table", "section_header", "text"}:
                out.append(span)
        return out
    except Exception:
        return []
