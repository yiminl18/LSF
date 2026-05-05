def rule_form_heading_before_company_name(doc: dict) -> list[dict]:
    """Match form headings that appear before the first large company-name cover header."""
    import re
    try:
        texts = doc.get("texts", [])
        company_idx = None
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and s.get("label") == "section_header":
                txt = (s.get("text") or "").strip()
                if txt and not re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", txt, re.I) and "SECURITIES AND EXCHANGE COMMISSION" not in txt.upper() and "CURRENT REPORT" not in txt.upper():
                    company_idx = i
                    break
        if company_idx is None:
            return []
        return [
            s for s in texts[:company_idx]
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (s.get("text") or ""), re.I)
            or "CURRENT REPORT" in ((s.get("text") or "").upper())
        ]
    except Exception:
        return []
