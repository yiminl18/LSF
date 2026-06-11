def rule_note_long_term_debt_headers(doc: dict) -> list[dict]:
    """Match section headers for notes titled long-term debt, debt, borrowings, or financing."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "")
            if re.search(r"\blong[- ]term debt\b", txt, re.I):
                out.append(span)
            elif re.search(r"\bdebt\b", txt, re.I) and not re.search(r"risk debt|troubled debt", txt, re.I):
                out.append(span)
            elif re.search(r"\bborrowings\b|\bfinancing\b|\bnotes payable\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
