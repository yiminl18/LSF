def rule_bestbuy_debt_table_or_note(doc: dict) -> list[dict]:
    """Match Best Buy debt tables or notes, especially long-term debt rows in balance sheet or debt footnotes."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            txt = (span.get("text") or "")
            if span.get("label") == "table":
                if "balance sheet" in path or "debt" in path or re.search(r"\blong[- ]term debt\b", txt, re.I):
                    out.append(span)
            else:
                if "best buy" in path and re.search(r"\blong[- ]term debt\b|\bdebt\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
