def rule_tables_with_assets_on_same_or_next_page_after_auditor(doc: dict) -> list[dict]:
    """Match asset tables on the same or next page after auditor-report language."""
    try:
        auditor_pages = set()
        for span in doc.get("texts", []):
            low = (span.get("text") or "").lower()
            if "independent registered public accounting firm" in low or "independent auditor" in low:
                auditor_pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in auditor_pages.union({p + 1 for p in auditor_pages}, {p + 2 for p in auditor_pages}):
                if "assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
