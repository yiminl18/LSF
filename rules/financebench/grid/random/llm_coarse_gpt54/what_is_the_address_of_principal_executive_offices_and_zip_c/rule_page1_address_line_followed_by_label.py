def rule_page1_address_line_followed_by_label(doc: dict) -> list[dict]:
    """Match page-1 spans whose next span is an address label, indicating the current span is the address."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            cur, nxt = texts[i], texts[i + 1]
            if cur.get("page_no") == 1 and nxt.get("page_no") == 1:
                if re.search(r'address of principal executive offices', (nxt.get("text") or ""), re.I):
                    out.append(cur)
        return out
    except Exception:
        return []
