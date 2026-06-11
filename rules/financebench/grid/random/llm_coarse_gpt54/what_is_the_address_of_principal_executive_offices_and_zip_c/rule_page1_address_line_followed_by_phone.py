def rule_page1_address_line_followed_by_phone(doc: dict) -> list[dict]:
    """Match page-1 spans immediately followed by a registrant telephone label or number."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            cur, nxt = texts[i], texts[i + 1]
            if cur.get("page_no") == 1 and nxt.get("page_no") == 1:
                tnext = (nxt.get("text") or "") + " " + (nxt.get("text_span") or "")
                if re.search(r'registrant.?s telephone number|\(\d{3}\)|\+\d{2}', tnext, re.I):
                    if re.search(r'\d{2,}.*(?:Street|Avenue|Boulevard|Drive|Road|Center|Plaza|California|Washington|New York|Bristol|United Kingdom|CA|WA)', (cur.get("text") or ""), re.I):
                        out.append(cur)
        return out
    except Exception:
        return []
