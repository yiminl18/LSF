def rule_form_code_with_annual_report_context(doc: dict) -> list[dict]:
    """Match 10-K form spans when ANNUAL REPORT appears in the same or nearby cover spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if re.search(r"\bFORM\s+10-K\b", span.get("text") or "", re.I):
                neighborhood = " ".join((texts[j].get("text") or "") for j in range(max(0, i-2), min(len(texts), i+3))).upper()
                if "ANNUAL REPORT" in neighborhood:
                    out.append(span)
        return out
    except Exception:
        return []
