def rule_10q_quarterly_combo(doc: dict) -> list[dict]:
    """Match both FORM 10-Q and QUARTERLY REPORT cues, useful for 10-Q filings."""
    import re
    try:
        out = []
        has_10q = any(re.search(r"\bFORM\s+10-Q\b", (s.get("text") or ""), re.I) for s in doc.get("texts", []))
        if not has_10q:
            return []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).upper()
            if "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt or re.search(r"\bFORM\s+10-Q\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
