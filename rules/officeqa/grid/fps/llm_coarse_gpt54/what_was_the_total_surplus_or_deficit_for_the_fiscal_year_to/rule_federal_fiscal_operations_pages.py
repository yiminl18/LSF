def rule_federal_fiscal_operations_pages(doc: dict) -> list[dict]:
    """Match table spans on pages headed Federal Fiscal Operations with fiscal-operation summary wording."""
    import re
    out = []
    try:
        page_has_header = {}
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'federal fiscal operations', txt, re.I):
                page_has_header[span.get("page_no")] = True
        for span in doc.get("texts", []):
            if span.get("label") == "table" and page_has_header.get(span.get("page_no")):
                txt = (span.get("text") or "")
                if re.search(r'(surplus|deficit|receipts|outlays)', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
