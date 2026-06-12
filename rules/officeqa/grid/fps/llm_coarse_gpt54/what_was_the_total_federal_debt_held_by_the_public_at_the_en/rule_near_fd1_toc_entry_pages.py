def rule_near_fd1_toc_entry_pages(doc: dict) -> list[dict]:
    """Match tables on pages referenced by nearby contents entries for FD-1 Summary of Federal Debt."""
    try:
        import re
        target_pages = set()
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            low = txt.lower()
            if "fd-1" in low and "summary of federal debt" in low:
                for m in re.finditer(r"\b(\d{1,3})\b", txt):
                    target_pages.add(int(m.group(1)))
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table" and int(span.get("page_no", -999)) in target_pages
        ]
    except Exception:
        return []
