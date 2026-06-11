def rule_form_code_with_fiscal_year_context(doc: dict) -> list[dict]:
    """Match 10-K-related spans when 'fiscal year ended' appears nearby."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = span.get("text") or ""
            if re.search(r"\bFORM\s+10-K\b", t, re.I) or "ANNUAL REPORT" in t.upper():
                neighborhood = " ".join((texts[j].get("text") or "") for j in range(max(0, i-3), min(len(texts), i+4))).upper()
                if "FISCAL YEAR ENDED" in neighborhood or "FOR THE FISCAL YEAR ENDED" in neighborhood:
                    out.append(span)
        return out
    except Exception:
        return []
