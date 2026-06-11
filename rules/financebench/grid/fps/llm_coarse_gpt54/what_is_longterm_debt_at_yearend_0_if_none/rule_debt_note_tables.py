def rule_debt_note_tables(doc: dict) -> list[dict]:
    """Match debt-related note tables under notes/footnotes sections mentioning debt, borrowings, notes, or financing."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            text = (span.get("text") or "").lower()
            if any(k in path for k in ["debt", "borrow", "notes payable", "long-term debt", "financing", "liquidity"]):
                out.append(span)
                continue
            if re.search(r"\blong[- ]term debt\b", text) or re.search(r"\bnotes due\b", text):
                out.append(span)
        return out
    except Exception:
        return []
