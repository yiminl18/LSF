def rule_tables_with_receipts_source_in_neighbor_text(doc: dict) -> list[dict]:
    """Match tables preceded by nearby text '(In millions of dollars)' and a receipts-by-source header."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            window = texts[max(0, i - 3):i]
            joined = "\n".join((s.get("text") or "") for s in window)
            if re.search(r'Budget Receipts by Source', joined, re.I) and re.search(r'In millions of dollars', joined, re.I):
                out.append(span)
    except Exception:
        return []
    return out
