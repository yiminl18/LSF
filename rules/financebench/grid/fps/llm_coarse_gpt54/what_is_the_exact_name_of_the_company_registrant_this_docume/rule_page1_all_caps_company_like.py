def rule_page1_all_caps_company_like(doc: dict) -> list[dict]:
    """Match all-caps page-1 spans that are likely company names and not SEC/form boilerplate."""
    try:
        texts = doc.get("texts", [])
        out = []
        bad = {"FORM 10-K", "FORM 10-Q", "FORM 8-K", "CURRENT REPORT", "PART I", "OR"}
        for span in texts:
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1 or not txt:
                continue
            if txt in bad:
                continue
            if "SECURITIES AND EXCHANGE COMMISSION" in txt or "UNITED STATES" in txt:
                continue
            if span.get("all_cap", 0) == 1 and len(txt) >= 5:
                if any(ch.isalpha() for ch in txt):
                    out.append(span)
        return out
    except Exception:
        return []
