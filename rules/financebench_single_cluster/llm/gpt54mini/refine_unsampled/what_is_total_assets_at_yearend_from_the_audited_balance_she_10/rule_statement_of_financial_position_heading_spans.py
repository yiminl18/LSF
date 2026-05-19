def rule_statement_of_financial_position_heading_spans(doc: dict) -> list[dict]:
    """Match section headers explicitly naming the statement of financial position."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").lower()
            if "statement of financial position" in text:
                out.append(span)
        return out
    except Exception:
        return []
