def rule_federal_debt_table_numbered_fd6_or_fd8_or_fd9(doc: dict) -> list[dict]:
    """Match Federal Debt tables whose title text includes FD/FO numbering associated with statutory limit tables."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"table", "section_header", "text"}:
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "federal debt" in path.lower() and re.search(r"\b(FD|FO)[-\s]?(6|8|9)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
