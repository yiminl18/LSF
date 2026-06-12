def rule_numeric_spans_on_currency_pages(doc: dict) -> list[dict]:
    """Return numeric-heavy spans on pages likely to contain the currency/coin table."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'currency\s+and\s+coin|per\s+capita|USCC-?2|C-?2', txt, re.I):
                pages.add(span.get("page_no"))
                for n in re.findall(r'\b\d{1,4}\b', txt):
                    try:
                        pages.add(int(n))
                    except Exception:
                        pass
        if not pages:
            return []
        for span in doc.get("texts", []):
            if span.get("page_no") not in pages:
                continue
            txt = (span.get("text") or "")
            if re.search(r'\b\d[\d,\.]*\b', txt):
                out.append(span)
    except Exception:
        return []
    return out
