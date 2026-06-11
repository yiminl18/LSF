def rule_balance_sheet_header_near_debt(doc: dict) -> list[dict]:
    """Match section headers for balance sheets and nearby tables/spans mentioning debt."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and "balance sheet" in (span.get("text") or "").lower():
                out.append(span)
                for j in range(i + 1, min(i + 6, len(texts))):
                    s2 = texts[j]
                    if re.search(r"\bdebt\b|\blong[- ]term debt\b", s2.get("text") or "", re.I) or s2.get("label") == "table":
                        out.append(s2)
        return out
    except Exception:
        return []
