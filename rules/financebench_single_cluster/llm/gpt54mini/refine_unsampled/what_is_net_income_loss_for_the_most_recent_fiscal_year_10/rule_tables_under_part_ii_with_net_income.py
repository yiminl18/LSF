def rule_tables_under_part_ii_with_net_income(doc: dict) -> list[dict]:
    """Match tables under Part II that mention net income/earnings/loss."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            path = (s.get("structure", {}) or {}).get("path_text", "") or ""
            txt = s.get("text", "") or ""
            if re.search(r"part ii", path, re.I) and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                out.append(s)
        return out
    except Exception:
        return []
