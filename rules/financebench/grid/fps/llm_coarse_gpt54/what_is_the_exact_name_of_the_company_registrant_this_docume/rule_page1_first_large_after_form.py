def rule_page1_first_large_after_form(doc: dict) -> list[dict]:
    """Match the first large span on page 1 after any FORM heading."""
    try:
        texts = doc.get("texts", [])
        seen_form = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").lower()
            if "form 10-k" in txt or "form 10-q" in txt or "form 8-k" in txt:
                seen_form = True
                continue
            if seen_form and float(span.get("size", 0) or 0) >= 12:
                if "current report" in txt or "securities and exchange commission" in txt:
                    continue
                return [span]
        return []
    except Exception:
        return []
