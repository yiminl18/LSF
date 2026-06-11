def rule_boeing_debt_narrative_or_table(doc: dict) -> list[dict]:
    """Match Boeing debt-related narrative or tables, often in liquidity/financial condition or debt notes."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if span.get("label") == "table":
                if "debt" in path or "balance sheet" in path or re.search(r"\blong[- ]term debt\b", txt, re.I):
                    out.append(span)
            else:
                if any(k in path for k in ["liquidity", "financial condition", "debt"]):
                    if re.search(r"\bdebt\b", txt, re.I):
                        out.append(span)
        return out
    except Exception:
        return []
