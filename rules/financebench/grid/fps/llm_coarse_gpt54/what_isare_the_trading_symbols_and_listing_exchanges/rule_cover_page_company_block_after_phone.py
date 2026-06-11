def rule_cover_page_company_block_after_phone(doc: dict) -> list[dict]:
    """Match spans after the registrant phone number on page 1, where symbol/exchange often appear."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            txt = (s.get("text") or "")
            if s.get("page_no") == 1 and re.search(r"telephone number|including area code|\(\d{3}\)\s*\d", txt, re.I):
                for j in range(i, min(len(texts), i + 15)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
