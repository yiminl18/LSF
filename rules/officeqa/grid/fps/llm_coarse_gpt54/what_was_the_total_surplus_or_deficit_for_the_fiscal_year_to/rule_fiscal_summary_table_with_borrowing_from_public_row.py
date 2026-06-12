def rule_fiscal_summary_table_with_borrowing_from_public_row(doc: dict) -> list[dict]:
    """Match tables with a row 'Borrowing from the public' near the deficit summary."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'borrowing from the public', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
