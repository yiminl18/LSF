def rule_note_long_term_debt_table(doc: dict) -> list[dict]:
    """Match tables in debt notes where long-term debt may be disclosed outside the balance sheet."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure") or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"note .*debt|long[\-\s]?term debt|borrowings", path + " " + text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
