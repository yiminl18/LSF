def rule_modern_federal_debt_contents_fd6(doc: dict) -> list[dict]:
    """Match modern contents entries where FD-6 is Debt Subject to Statutory Limit."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "contents" in path.lower() and re.search(r"\bFD[-\s]?6\b", text, re.I) and re.search(r"debt subject to statutory", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
