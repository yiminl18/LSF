def rule_selected_financial_data_net_income_table(doc: dict) -> list[dict]:
    """Match Selected Financial Data tables containing net income/earnings rows."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            txt = span.get("text", "") or ""
            if not re.search(r"selected (financial data|consolidated financial data)", path + " " + txt, re.I):
                continue
            if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
