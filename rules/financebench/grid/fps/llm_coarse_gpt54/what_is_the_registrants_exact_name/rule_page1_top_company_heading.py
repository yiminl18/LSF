def rule_page1_top_company_heading(doc: dict) -> list[dict]:
    """Match prominent page-1 company-name headings near the top, excluding SEC/FORM headings."""
    try:
        texts = doc.get("texts", [])
        out = []
        bad = [
            "securities and exchange commission",
            "united states",
            "form 10-k",
            "form 10-q",
            "form 8-k",
            "current report",
            "annual report",
            "quarterly report",
            "transition report",
            "washington, d.c.",
            "washington, dc",
        ]
        for i, span in enumerate(texts[:40]):
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if not txt:
                continue
            if any(b in low for b in bad):
                continue
            if span.get("bold") != 1:
                continue
            if float(span.get("size") or 0) < 12:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            if len(txt) > 120:
                continue
            out.append(span)
        return out
    except Exception:
        return []
