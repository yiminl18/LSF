def rule_text_spans_with_net_receipts_reflected(doc: dict) -> list[dict]:
    """Match prose spans where net receipts are described for individual income taxes."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") in {"text", "list_item"}:
                if re.search(r'Individual income taxes', txt, re.I) and re.search(r'net receipts|increase in net receipts|decrease in net receipts', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
