def rule_late_era_contents_fd5_page(doc: dict) -> list[dict]:
    """Use late-era contents entry for FD-5 and return spans on that exact page."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "contents" in path and "fd-5" in txt and "average length" in txt:
                nums = re.findall(r'\b\d{1,3}\b', txt)
                for n in nums[-1:]:
                    pages.add(int(n))
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
