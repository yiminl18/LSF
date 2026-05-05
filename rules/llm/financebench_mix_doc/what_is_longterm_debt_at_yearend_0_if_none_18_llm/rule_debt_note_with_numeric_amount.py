def rule_debt_note_with_numeric_amount(doc: dict) -> list[dict]:
    """Match debt note sections containing numeric debt amounts."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r"debt|borrowings|credit facilities", txt, re.I) or re.search(r"debt|borrowings|credit facilities", path, re.I):
                if re.search(r"\$?\s*\(?\d[\d,]*(\.\d+)?\)?", txt):
                    out.append(span)
    except Exception:
        return []
    return out
