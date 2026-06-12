def rule_fd5_private_investors_phrase(doc: dict) -> list[dict]:
    """Match spans containing the modern phrasing 'held by private investors' or 'private investors' near average length."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if (
                "average length" in txt
                and "private investors" in txt
                and "marketable" in txt
                and "debt" in txt
            ):
                out.append(span)
    except Exception:
        return []
    return out
