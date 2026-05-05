def rule_balance_sheet_heading_spans(doc: dict) -> list[dict]:
    """Match section headers explicitly naming the balance sheet."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").lower()
            if "balance sheet" in text or "balance sheets" in text:
                out.append(span)
        return out
    except Exception:
        return []
