def rule_form_text_anywhere(doc: dict) -> list[dict]:
    """Match any span whose text contains a common SEC form label such as FORM 10-K, FORM 10-Q, or FORM 8-K."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
