def rule_coverpage_text_with_exact_name_and_identifiers(doc: dict) -> list[dict]:
    """Match spans mentioning exact-name charter text together with nearby cover identifiers."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if re.search(r"exact name of registrant|exact name of registrant as specified in (its )?charter", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
