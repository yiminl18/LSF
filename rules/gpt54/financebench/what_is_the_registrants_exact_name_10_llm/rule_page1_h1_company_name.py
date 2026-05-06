def rule_page1_h1_company_name(doc: dict) -> list[dict]:
    """Match page-1 H1/section_header spans that look like the registrant name block."""
    try:
        texts = doc.get("texts", [])
        out = []
        bad = [
            "united states securities and exchange commission",
            "form 10-k",
            "or",
            "documents incorporated by reference",
            "part i",
        ]
        for span in texts:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            level = span.get("structure", {}).get("level")
            if level not in {"H1", "H2"}:
                continue
            if any(b == low for b in bad):
                continue
            if "exact name of registrant" in low:
                continue
            if len(txt) < 3:
                continue
            if any(ch.isalpha() for ch in txt):
                out.append(span)
        return out
    except Exception:
        return []
