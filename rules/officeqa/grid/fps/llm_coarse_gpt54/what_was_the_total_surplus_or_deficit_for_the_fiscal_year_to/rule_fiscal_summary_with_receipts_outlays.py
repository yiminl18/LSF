def rule_fiscal_summary_with_receipts_outlays(doc: dict) -> list[dict]:
    """Match tables containing receipts, outlays, and total surplus/deficit together."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if (
                re.search(r'(net )?receipts', txt, re.I)
                and re.search(r'(net )?outlays', txt, re.I)
                and re.search(r'total surplus.*deficit', txt, re.I)
            ):
                out.append(span)
    except Exception:
        return []
    return out
