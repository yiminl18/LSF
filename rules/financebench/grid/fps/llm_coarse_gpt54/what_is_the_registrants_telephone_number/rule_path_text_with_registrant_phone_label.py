def rule_path_text_with_registrant_phone_label(doc: dict) -> list[dict]:
    """Match spans whose path_text contains the registrant telephone label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number", path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
