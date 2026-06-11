def rule_page1_h1_or_h2_company_block(doc: dict) -> list[dict]:
    """Match page-1 H1/H2 spans that look like the registrant block rather than SEC/form headers."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            lvl = span.get("structure", {}).get("level")
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if lvl not in {"H1", "H2"}:
                continue
            if "securities and exchange commission" in txt or "form 10-" in txt or "form 8-k" in txt or "current report" in txt:
                continue
            if "exact name of registrant" in txt:
                continue
            out.append(span)
        return out
    except Exception:
        return []
