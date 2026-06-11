def rule_form_code_h1_or_h2(doc: dict) -> list[dict]:
    """Match H1/H2 section headers containing a common form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            lvl = ((span.get("structure") or {}).get("level") or "")
            if span.get("label") == "section_header" and lvl in {"H1", "H2"}:
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I):
                    out.append(span)
        return out
    except Exception:
        return []
