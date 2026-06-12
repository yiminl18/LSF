def rule_federal_fiscal_operations_pages(doc: dict) -> list[dict]:
    """Match table spans on pages headed Federal Fiscal Operations that mention receipts/outlays/deficit."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                re.search(r'Federal Fiscal Operations', path, re.I)
                and (
                    re.search(r'Fiscal year', txt, re.I)
                    or re.search(r'Net receipts|Total receipts', txt, re.I)
                    or re.search(r'Net outlays|Total outlays', txt, re.I)
                    or re.search(r'Fiscal .* to date', txt, re.I)
                )
            ):
                out.append(span)
        return out
    except Exception:
        return []
