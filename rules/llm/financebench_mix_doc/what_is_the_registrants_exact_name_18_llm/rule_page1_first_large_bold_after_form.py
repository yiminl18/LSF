def rule_page1_first_large_bold_after_form(doc: dict) -> list[dict]:
    """Match the first large bold span on page 1 after a form header, excluding boilerplate."""
    try:
        texts = doc.get("texts", [])
        seen_form = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").lower()
            if "form 10-" in txt or "form 8-k" in txt:
                seen_form = True
                continue
            if (
                seen_form
                and span.get("bold") == 1
                and float(span.get("size", 0) or 0) >= 10
                and "securities and exchange commission" not in txt
                and "current report" not in txt
                and "washington, d.c." not in txt
                and "commission file" not in txt
                and "annual report pursuant" not in txt
                and "quarterly report pursuant" not in txt
                and "transition report pursuant" not in txt
            ):
                return [span]
        return []
    except Exception:
        return []
