def rule_address_before_label(doc: dict) -> list[dict]:
    """Return spans near the Address of principal executive offices label on page 1."""
    try:
        texts = doc.get("texts", [])
        for i, s in enumerate(texts):
            if s.get("page_no") != 1:
                continue
            text_lower = s.get("text", "").lower()
            if "address of principal" in text_lower or "principal executive office" in text_lower:
                start = max(0, i - 4)
                return texts[start:i+1]
        return []
    except Exception:
        return []
