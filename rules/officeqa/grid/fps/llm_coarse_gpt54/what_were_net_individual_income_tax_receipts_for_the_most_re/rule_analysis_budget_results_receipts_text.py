def rule_analysis_budget_results_receipts_text(doc: dict) -> list[dict]:
    """Match analysis prose under budget results sections that discusses receipts by source and individual income taxes."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if span.get("label") in {"text", "list_item"}:
                if re.search(r'Budget results', path, re.I) or re.search(r'Fourth-Quarter Receipts|Receipts', path, re.I):
                    if re.search(r'Individual income taxes', txt, re.I):
                        out.append(span)
    except Exception:
        return []
    return out
