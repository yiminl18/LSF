def rule_debt_note_section_header(doc: dict) -> list[dict]:
    """Match section headers or nearby spans for debt notes, useful when answer is in a debt footnote table."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                text = span.get("text", "") or ""
                path = (span.get("structure") or {}).get("path_text", "") or ""
                if re.search(r"\bdebt\b|\blong[\-\s]?term debt\b|\blong[\-\s]?term borrowings\b", text + " " + path, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
