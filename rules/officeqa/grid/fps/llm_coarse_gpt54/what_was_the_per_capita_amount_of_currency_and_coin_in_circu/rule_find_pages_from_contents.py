def rule_find_pages_from_contents(doc: dict) -> list[dict]:
    """Find the actual table page by reading contents entries for currency/coin tables and returning spans on those pages."""
    import re
    out = []
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'(currency\s+and\s+coin\s+in\s+circulation|currency\s+and\s+coin\s+outstanding\s+and\s+in\s+circulation|amounts\s+outstanding\s+and\s+in\s+circulation)', txt, re.I):
                nums = re.findall(r'\b\d{1,4}\b', txt)
                for n in nums:
                    try:
                        val = int(n)
                        if 1 <= val <= 10000:
                            target_pages.add(val)
                    except Exception:
                        pass
        if not target_pages:
            return []
        for span in doc.get("texts", []):
            if span.get("page_no") in target_pages:
                out.append(span)
    except Exception:
        return []
    return out
