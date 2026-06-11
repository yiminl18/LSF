def rule_form_code_with_report_keyword_nearby(doc: dict) -> list[dict]:
    """Match spans containing a form code when adjacent spans mention annual/quarterly/current report."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = span.get("text") or ""
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", t, re.I):
                neighborhood = " ".join(
                    (texts[j].get("text") or "")
                    for j in range(max(0, i - 2), min(len(texts), i + 3))
                ).upper()
                if any(k in neighborhood for k in ["ANNUAL REPORT", "QUARTERLY REPORT", "CURRENT REPORT"]):
                    out.append(span)
        return out
    except Exception:
        return []
