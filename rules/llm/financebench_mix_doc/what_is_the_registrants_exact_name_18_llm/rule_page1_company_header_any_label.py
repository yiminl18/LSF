def rule_page1_company_header_any_label(doc: dict) -> list[dict]:
    """Match any page-1 bold large span that looks like the registrant name, regardless of label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and float(span.get("size", 0) or 0) >= 10
                and re.search(r"[A-Za-z]", txt)
                and "form 10-" not in low
                and "form 8-k" not in low
                and "current report" not in low
                and "securities and exchange commission" not in low
                and "washington, d.c." not in low
                and "commission file" not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
