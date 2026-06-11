def rule_page1_large_bold_company_name(doc: dict) -> list[dict]:
    """Match large bold spans on page 1 that look like the registrant name near the cover."""
    try:
        texts = doc.get("texts", [])
        out = []
        bad = [
            "form 10-k", "form 10-q", "form 8-k", "current report",
            "united states", "securities and exchange commission",
            "washington, d.c.", "part i", "news release", "documents incorporated by reference"
        ]
        for span in texts:
            txt = (span.get("text", "") or "").strip()
            if not txt or span.get("page_no") != 1:
                continue
            if span.get("bold", 0) != 1:
                continue
            if float(span.get("size", 0) or 0) < 12:
                continue
            low = txt.lower()
            if any(b in low for b in bad):
                continue
            if len(txt) < 3:
                continue
            out.append(span)
        return out
    except Exception:
        return []
