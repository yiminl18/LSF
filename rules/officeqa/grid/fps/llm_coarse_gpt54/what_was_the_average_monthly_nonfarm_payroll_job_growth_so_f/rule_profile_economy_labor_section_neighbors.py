def rule_profile_economy_labor_section_neighbors(doc: dict) -> list[dict]:
    """Return text spans near labor-related section headers inside Profile of the Economy."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header":
                hdr = span.get("text") or ""
                if re.search(r'(Employment and unemployment|Labor Markets and Wages|Labor Markets)', hdr, re.I):
                    for j in range(i + 1, min(i + 6, len(texts))):
                        s = texts[j]
                        if s.get("label") == "text":
                            out.append(s)
        return out
    except Exception:
        return []
