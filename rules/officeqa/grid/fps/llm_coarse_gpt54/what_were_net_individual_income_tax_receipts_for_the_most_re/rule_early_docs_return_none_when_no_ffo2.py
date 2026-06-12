def rule_early_docs_return_none_when_no_ffo2(doc: dict) -> list[dict]:
    """Conservative rule: if no FFO-2/receipts-by-source evidence exists, return no spans."""
    import re
    try:
        texts = doc.get("texts", [])
        found = False
        for span in texts:
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'FFO-2|Budget Receipts by Source|On-Budget and Off-Budget Receipts by Source', txt, re.I):
                found = True
                break
            if re.search(r'FFO-2|Budget Receipts by Source', path, re.I):
                found = True
                break
        return [] if not found else []
    except Exception:
        return []
