def rule_quarter_fiscal_exact(doc: dict) -> list[dict]:
    """Match exact quarter/fiscal spans like 'FIRST QUARTER, FISCAL 1985'."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(FIRST|SECOND|THIRD|FOURTH|1ST|2ND|3RD|4TH)\s+QUARTER,?\s+FISCAL\s+\d{4}$",
            re.I,
        )
        return [s for s in texts if pat.match((s.get("text") or "").strip())]
    except Exception:
        return []
