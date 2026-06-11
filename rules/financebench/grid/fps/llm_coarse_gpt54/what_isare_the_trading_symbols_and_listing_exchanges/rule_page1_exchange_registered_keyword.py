def rule_page1_exchange_registered_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Name of each exchange on which registered'."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and re.search(r"Name of each exchange on which registered", (s.get("text") or ""), re.I)
        ]
    except Exception:
        return []
