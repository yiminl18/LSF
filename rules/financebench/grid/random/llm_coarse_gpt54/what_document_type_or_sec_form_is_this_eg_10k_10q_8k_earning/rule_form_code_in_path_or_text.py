def rule_form_code_in_path_or_text(doc: dict) -> list[dict]:
    """Match spans where either text or path_text contains a common SEC form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = span.get("text") or ""
            p = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", t, re.I) or re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", p, re.I):
                out.append(span)
        return out
    except Exception:
        return []
