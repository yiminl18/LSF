def rule_fiscal_operations_page_range(doc: dict) -> list[dict]:
    """Match likely answer tables on early Federal Fiscal Operations pages (roughly pages 15-25 in older issues)."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if span.get("label") == "table" and isinstance(p, int) and 15 <= p <= 25:
                txt = (span.get("text") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                if (
                    re.search(r'summary of fiscal operations', txt, re.I)
                    or re.search(r'summary of fiscal operations', path, re.I)
                    or re.search(r'total surplus.*deficit', txt, re.I)
                ):
                    out.append(span)
    except Exception:
        return []
    return out
