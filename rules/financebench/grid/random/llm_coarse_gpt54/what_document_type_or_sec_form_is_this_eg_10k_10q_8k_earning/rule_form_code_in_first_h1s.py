def rule_form_code_in_first_h1s(doc: dict) -> list[dict]:
    """Match the first few H1-like section headers containing a form code."""
    import re
    try:
        out = []
        count = 0
        for span in doc.get("texts", []):
            if span.get("label") == "section_header" and (span.get("structure") or {}).get("level") == "H1":
                count += 1
                if count <= 5 and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I):
                    out.append(span)
        return out
    except Exception:
        return []
