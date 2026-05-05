def rule_cover_page_report_type_keywords(doc: dict) -> list[dict]:
    """Match page-1 cover-page spans with strong report-type keywords."""
    import re
    try:
        pats = [
            r"\bFORM\s+10-K\b",
            r"\bFORM\s+10-Q\b",
            r"\bFORM\s+8-K\b",
            r"\bCURRENT REPORT\b",
            r"\bANNUAL REPORT PURSUANT TO SECTION 13 OR 15\(D\)\b",
            r"\bQUARTERLY REPORT PURSUANT TO SECTION 13 OR 15\(D\)\b",
        ]
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if any(re.search(p, text, re.I) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
