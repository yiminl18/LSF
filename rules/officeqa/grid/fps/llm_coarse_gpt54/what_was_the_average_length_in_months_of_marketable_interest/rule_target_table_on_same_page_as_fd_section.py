def rule_target_table_on_same_page_as_fd_section(doc: dict) -> list[dict]:
    """Return table spans on pages that also contain a Federal Debt section header."""
    out = []
    try:
        fd_pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if span.get("label") == "section_header" and "federal debt" in txt:
                if span.get("page_no") is not None:
                    fd_pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in fd_pages:
                cells = (((span.get("table_data") or {}).get("cells")) or [])
                joined = " ".join((c.get("text") or "") for c in cells).lower()
                if "average length" in joined or "maturity distribution" in joined:
                    out.append(span)
    except Exception:
        return []
    return out
