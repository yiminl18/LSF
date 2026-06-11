def rule_page1_inline_company_block_with_state_value_and_ein(doc: dict) -> list[dict]:
    """Match page-1 spans containing both an EIN pattern and a likely state/jurisdiction phrase/value."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") != 1:
                continue
            if re.search(r"\b\d{2}-\d{7}\b", text):
                if re.search(r"\bDelaware\b|\bWashington\b|\bNew York\b|\bJersey\b|\bCalifornia\b|\bVirginia\b|\bNevada\b", text, re.I):
                    out.append(span)
                elif re.search(r"State or other jurisdiction", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
