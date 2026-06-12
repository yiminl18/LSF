def rule_ofs2_estimated_ownership_private_investors(doc: dict) -> list[dict]:
    """Match OFS-2 contents/table spans about estimated ownership by private investors, a nearby supporting section in later documents."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "estimated ownership" in txt and "private investors" in txt:
                out.append(span)
    except Exception:
        return []
    return out
