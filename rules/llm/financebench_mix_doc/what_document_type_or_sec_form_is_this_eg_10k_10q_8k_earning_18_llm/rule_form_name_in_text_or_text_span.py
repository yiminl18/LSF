def rule_form_name_in_text_or_text_span(doc: dict) -> list[dict]:
    """Match spans where either text or text_span contains a form name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            blob = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
