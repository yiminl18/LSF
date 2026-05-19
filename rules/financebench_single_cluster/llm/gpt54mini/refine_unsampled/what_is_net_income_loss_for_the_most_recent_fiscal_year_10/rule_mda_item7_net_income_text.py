def rule_mda_item7_net_income_text(doc: dict) -> list[dict]:
    """Match Item 7 spans mentioning net income/loss/earnings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            txt = span.get("text", "") or ""
            if re.search(r"item\s*7", path, re.I) and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
