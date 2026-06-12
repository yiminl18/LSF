def rule_federal_debt_section_spans(doc: dict) -> list[dict]:
    """Match spans under the Federal Debt section that mention average length and marketable debt."""
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                "federal debt" in path
                and "average length" in txt
                and "marketable" in txt
            ):
                out.append(span)
    except Exception:
        return []
    return out
