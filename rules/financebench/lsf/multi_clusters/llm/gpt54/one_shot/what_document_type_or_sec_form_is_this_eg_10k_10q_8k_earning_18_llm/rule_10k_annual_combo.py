def rule_10k_annual_combo(doc: dict) -> list[dict]:
    """Match both FORM 10-K and ANNUAL REPORT cues, useful for 10-K filings."""
    import re
    try:
        out = []
        has_10k = any(re.search(r"\bFORM\s+10-K\b", (s.get("text") or ""), re.I) for s in doc.get("texts", []))
        if not has_10k:
            return []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).upper()
            if "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt or re.search(r"\bFORM\s+10-K\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
