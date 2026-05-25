def rule_tables_with_balance_sheet_in_toc_path(doc: dict) -> list[dict]:
    """Match tables whose path or nearby context suggests they are listed in the table of contents as balance sheet."""
    try:
        out = []
        texts = doc.get("texts", [])
        toc_pages = set()
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "table of contents" in path or path.endswith("| index") or "index to the form 10-k" in path:
                if "balance sheet" in (span.get("text") or "").lower():
                    toc_pages.add(span.get("page_no"))
        if not toc_pages:
            return out
        for span in texts:
            if span.get("label") == "table" and span.get("page_no", 0) > max(toc_pages):
                txt = (span.get("text") or "").lower()
                if "assets" in txt and ("liabilities" in txt or "equity" in txt):
                    out.append(span)
        return out
    except Exception:
        return []
