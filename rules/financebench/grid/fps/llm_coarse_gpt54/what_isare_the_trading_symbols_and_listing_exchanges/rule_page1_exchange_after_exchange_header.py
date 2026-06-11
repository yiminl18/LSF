def rule_page1_exchange_after_exchange_header(doc: dict) -> list[dict]:
    """Match spans immediately following an exchange-header cue on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and re.search(r"Name of each exchange on which registered", (s.get("text") or ""), re.I):
                for j in range(i + 1, min(len(texts), i + 4)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
