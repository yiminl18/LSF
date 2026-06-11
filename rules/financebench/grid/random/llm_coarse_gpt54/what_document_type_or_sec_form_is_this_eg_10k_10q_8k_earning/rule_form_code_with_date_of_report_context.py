def rule_form_code_with_date_of_report_context(doc: dict) -> list[dict]:
    """Match 8-K-related spans when 'Date of Report' appears nearby."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = span.get("text") or ""
            if re.search(r"\bFORM\s+8-K\b", t, re.I) or "CURRENT REPORT" in t.upper():
                neighborhood = " ".join((texts[j].get("text") or "") for j in range(max(0, i-3), min(len(texts), i+4))).upper()
                if "DATE OF REPORT" in neighborhood:
                    out.append(span)
        return out
    except Exception:
        return []
