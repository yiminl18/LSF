def rule_federal_debt_pages_20_to_40(doc: dict) -> list[dict]:
    """Match spans on likely Federal Debt data pages where statutory limit tables usually appear in these bulletins."""
    try:
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if isinstance(p, int) and 20 <= p <= 40:
                path = ((span.get("structure") or {}).get("path_text") or "")
                text = (span.get("text") or "")
                if "federal debt" in path.lower() or "federal debt" in text.lower():
                    out.append(span)
        return out
    except Exception:
        return []
