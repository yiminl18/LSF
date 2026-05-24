def rule_any_financial_table_with_net_row(doc: dict) -> list[dict]:
    """Broad recall rule: any table that looks financial and contains a net income/earnings/loss row."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            txt = s.get("text", "") or ""
            if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I) and re.search(r"\$|20\d{2}|million|billion|consolidated|income|operations", txt, re.I):
                out.append(s)
        return out
    except Exception:
        return []
