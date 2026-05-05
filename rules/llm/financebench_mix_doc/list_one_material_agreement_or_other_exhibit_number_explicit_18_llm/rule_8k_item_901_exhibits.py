def rule_8k_item_901_exhibits(doc: dict) -> list[dict]:
    """Match spans under Item 9.01 Financial Statements and Exhibits."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if re.search(r"\bitem\s*9\.?01\b", txt, re.I) and re.search(r"financial statements and exhibits", txt, re.I):
                out.append(span)
            elif "item 9.01" in path.lower() and "exhibit" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
