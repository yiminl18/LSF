def rule_currency_coin_pages_from_contents_numbers(doc: dict) -> list[dict]:
    """Infer target pages from contents lines and return spans on those pages for currency/coin entries."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'currency\s+and\s+coin', txt, re.I):
                nums = re.findall(r'\b\d{1,4}\b', txt)
                if nums:
                    try:
                        pages.add(int(nums[-1]))
                    except Exception:
                        pass
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
