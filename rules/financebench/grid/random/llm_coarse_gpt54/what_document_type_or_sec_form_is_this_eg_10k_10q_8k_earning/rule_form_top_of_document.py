def rule_form_top_of_document(doc: dict) -> list[dict]:
    """Match early-document spans containing FORM 10-K/10-Q/8-K, favoring the first 15 spans."""
    import re
    try:
        out = []
        for span in (doc.get("texts", [])[:15]):
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I):
                out.append(span)
        return out
    except Exception:
        return []
