def rule_consolidated_results_table(doc: dict) -> list[dict]:
    """Match tables under paths mentioning consolidated financial statements/results with net income row."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            path = (s.get("structure", {}) or {}).get("path_text", "") or ""
            txt = s.get("text", "") or ""
            if re.search(r"consolidated|financial statements|results of operations", path + " " + txt, re.I) and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                out.append(s)
        return out
    except Exception:
        return []
