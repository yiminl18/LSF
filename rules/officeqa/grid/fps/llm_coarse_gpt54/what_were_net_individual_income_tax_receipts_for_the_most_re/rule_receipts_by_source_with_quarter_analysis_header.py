def rule_receipts_by_source_with_quarter_analysis_header(doc: dict) -> list[dict]:
    """Match quarter-analysis headers that often precede answer-bearing prose or tables."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "section_header":
                if re.search(r'Fourth-Quarter Receipts|First-Quarter Receipts|Budget results.*quarter', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
