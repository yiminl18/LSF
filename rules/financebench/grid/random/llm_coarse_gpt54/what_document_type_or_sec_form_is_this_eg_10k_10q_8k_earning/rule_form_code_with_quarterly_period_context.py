def rule_form_code_with_quarterly_period_context(doc: dict) -> list[dict]:
    """Match 10-Q-related spans when 'quarterly period ended' appears nearby."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = span.get("text") or ""
            if re.search(r"\bFORM\s+10-Q\b", t, re.I) or "QUARTERLY REPORT" in t.upper():
                neighborhood = " ".join((texts[j].get("text") or "") for j in range(max(0, i-3), min(len(texts), i+4))).upper()
                if "QUARTERLY PERIOD ENDED" in neighborhood or "FOR THE QUARTERLY PERIOD ENDED" in neighborhood:
                    out.append(span)
        return out
    except Exception:
        return []
