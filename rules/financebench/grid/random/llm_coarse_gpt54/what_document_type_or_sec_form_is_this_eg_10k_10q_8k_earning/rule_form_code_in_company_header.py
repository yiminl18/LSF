def rule_form_code_in_company_header(doc: dict) -> list[dict]:
    """Match company-name cover spans whose text_span contains a form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            ts = span.get("text_span") or ""
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", ts, re.I):
                out.append(span)
        return out
    except Exception:
        return []
