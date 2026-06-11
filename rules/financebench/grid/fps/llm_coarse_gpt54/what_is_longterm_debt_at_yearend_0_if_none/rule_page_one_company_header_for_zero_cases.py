def rule_page_one_company_header_for_zero_cases(doc: dict) -> list[dict]:
    """Match page-1 company header spans in short 8-K event filings where answer is likely 0."""
    try:
        out = []
        texts = doc.get("texts", [])
        has_8k = any("form 8-k" in (s.get("text") or "").lower() for s in texts)
        if not has_8k:
            return []
        for s in texts:
            if s.get("page_no") == 1 and s.get("label") == "section_header":
                txt = (s.get("text") or "")
                if any(k in txt.lower() for k in ["inc.", "corporation", "plc", "johnson & johnson", "ebay", "block", "best buy", "costco", "amcor"]):
                    out.append(s)
        return out
    except Exception:
        return []
