def rule_form_with_commission_file_number(doc: dict) -> list[dict]:
    """Match spans containing a form code and nearby Commission File Number text in the same span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = span.get("text") or ""
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", t, re.I) and re.search(r"Commission\s+File\s+Number|Commission\s+file\s+number|Commission File No\.", t, re.I):
                out.append(span)
        return out
    except Exception:
        return []
