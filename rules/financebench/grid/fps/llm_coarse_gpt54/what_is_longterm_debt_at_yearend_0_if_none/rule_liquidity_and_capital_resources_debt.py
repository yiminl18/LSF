def rule_liquidity_and_capital_resources_debt(doc: dict) -> list[dict]:
    """Match spans under liquidity/capital resources that mention debt balances."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if any(k in path for k in ["liquidity", "capital resources", "financial condition"]):
                if re.search(r"\bdebt\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
