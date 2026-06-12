def rule_statutory_limit_table_or_text_on_fd_pages(doc: dict) -> list[dict]:
    """Match table/text spans on Federal Debt pages that mention statutory limit concepts or likely answer amounts."""
    import re
    try:
        fd_pages = set()
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "federal debt" in text or "federal debt" in path:
                fd_pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") not in fd_pages:
                continue
            text = (span.get("text") or "")
            if re.search(r"statutory|debt subject|debt ceiling|debt limit|\$[\d,]+|\b\d{1,3}(?:,\d{3})+\s*(million|billion|trillion)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
