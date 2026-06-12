def rule_total_receipts_phrase_any_span(doc: dict) -> list[dict]:
    """Match any span containing Total receipts or Net receipts phrasing."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r'(Total receipts|Net receipts|Net budget receipts)', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
