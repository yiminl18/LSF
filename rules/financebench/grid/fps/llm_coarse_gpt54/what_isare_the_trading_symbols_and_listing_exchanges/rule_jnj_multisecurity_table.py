def rule_jnj_multisecurity_table(doc: dict) -> list[dict]:
    """Match Johnson & Johnson-style page-1 multi-row securities table with title, symbol, and exchange."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table" or s.get("page_no") != 1:
                continue
            cells = (((s.get("table_data") or {}).get("cells")) or [])
            joined = " | ".join((c.get("text") or "") for c in cells)
            if re.search(r"Trading Symbol\(s\)", joined, re.I) and re.search(r"New York Stock Exchange", joined, re.I):
                out.append(s)
        return out
    except Exception:
        return []
