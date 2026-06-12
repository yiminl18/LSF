def rule_page_near_contents_target_page(doc: dict) -> list[dict]:
    """Use contents entry page number for the target table and return spans on that page and the next page."""
    import re
    out = []
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "contents" in path and "maturity distribution" in txt and "average length" in txt:
                nums = re.findall(r'\b\d{1,3}\b', txt)
                for n in nums[-2:]:
                    try:
                        target_pages.add(int(n))
                    except Exception:
                        pass
        if not target_pages:
            return []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if any(p in (tp, tp + 1) for tp in target_pages):
                out.append(span)
    except Exception:
        return []
    return out
