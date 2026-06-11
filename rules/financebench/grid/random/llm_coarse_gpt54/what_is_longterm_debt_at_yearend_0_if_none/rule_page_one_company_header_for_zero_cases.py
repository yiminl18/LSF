def rule_page_one_company_header_for_zero_cases(doc: dict) -> list[dict]:
    """Match issuer/company H1 on page 1 for forms like 8-K where answer should default to 0 due to no debt disclosure."""
    try:
        out = []
        has_8k = any("8-k" in ((s.get("text", "") or "").lower()) for s in doc.get("texts", []))
        if not has_8k:
            return []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                lvl = ((span.get("structure") or {}).get("level") or "")
                if lvl == "H1":
                    out.append(span)
        return out
    except Exception:
        return []
