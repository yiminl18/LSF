def rule_page1_first_non_boilerplate_with_company_suffix(doc: dict) -> list[dict]:
    """Match the first page-1 non-boilerplate span containing a company suffix."""
    try:
        import re
        texts = doc.get("texts", [])
        pat = re.compile(r'\b(inc\.?|incorporated|corporation|company|plc|co\.)\b', re.I)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if not txt:
                continue
            if any(x in low for x in ["securities and exchange commission", "united states", "form 10-k", "form 10-q", "form 8-k", "current report"]):
                continue
            if pat.search(txt):
                return [span]
        return []
    except Exception:
        return []
