def rule_8k_current_report_combo(doc: dict) -> list[dict]:
    """Match both FORM 8-K and CURRENT REPORT cues, useful for 8-K filings."""
    import re
    try:
        out = []
        has_8k = any(re.search(r"\bFORM\s+8-K\b", (s.get("text") or ""), re.I) for s in doc.get("texts", []))
        if not has_8k:
            return []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).upper()
            if "CURRENT REPORT" in txt or re.search(r"\bFORM\s+8-K\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
