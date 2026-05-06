def rule_address_label_context(doc: dict) -> list[dict]:
    """Return spans near the Address of principal executive offices label on page 1-2."""
    try:
        texts = doc.get("texts", [])
        for i, s in enumerate(texts):
            if s.get("page_no", 0) > 2:
                continue
            text_lower = s.get("text", "").lower()
            if "address of principal" in text_lower or "principal executive office" in text_lower:
                start = max(0, i - 5)
                end = min(len(texts), i + 5)
                return texts[start:end]
        return []
    except Exception:
        return []
