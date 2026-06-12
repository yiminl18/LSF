def rule_text_spans_with_individual_income_tax_receipts_phrase(doc: dict) -> list[dict]:
    """Match prose spans explicitly saying individual income tax receipts."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "list_item"}:
                continue
            txt = (span.get("text") or "")
            if re.search(r'individual income tax receipts', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
