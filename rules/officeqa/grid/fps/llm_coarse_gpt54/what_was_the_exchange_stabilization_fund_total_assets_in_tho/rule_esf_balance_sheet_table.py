def rule_esf_balance_sheet_table(doc: dict) -> list[dict]:
    """Match table spans mentioning balance sheet in ESF context."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            combo = f"{path}\n{text}"
            if re.search(r'Balance sheet', combo, re.I) and re.search(r'ESF|Exchange Stabilization Fund', combo, re.I):
                out.append(span)
        return out
    except Exception:
        return []
