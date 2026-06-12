def rule_federal_debt_statutory_limit_headers(doc: dict) -> list[dict]:
    """Match section headers in or near the Federal Debt section that mention statutory limit/limitation."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if ("federal debt" in path.lower() or "federal debt" in text.lower()) and re.search(r"statutory (limit|limitation)|debt subject to statutory", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
