def rule_page1_top_bold_non_boilerplate(doc: dict) -> list[dict]:
    """Match top-of-page-1 bold spans with alphabetic text excluding SEC/form boilerplate."""
    try:
        texts = doc.get("texts", [])
        out = []
        boiler = [
            "united states securities and exchange commission",
            "washington, d.c. 20549",
            "form 10-k",
            "annual report pursuant",
            "transition report pursuant",
            "commission file number",
            "documents incorporated by reference",
            "part i",
            "table of contents",
            "index",
        ]
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if not any(ch.isalpha() for ch in txt):
                continue
            if any(b in low for b in boiler):
                continue
            if "exact name of registrant" in low:
                continue
            if (span.get("size") or 0) >= 7:
                out.append(span)
        return out
    except Exception:
        return []
