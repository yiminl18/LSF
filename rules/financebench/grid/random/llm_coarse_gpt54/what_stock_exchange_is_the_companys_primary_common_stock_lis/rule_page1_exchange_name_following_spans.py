def rule_page1_exchange_name_following_spans(doc: dict) -> list[dict]:
    """Return page-1 spans immediately following a 'Name of each exchange on which registered' label span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "name of each exchange on which registered" in txt:
                for j in range(i + 1, min(i + 6, len(texts))):
                    nxt = texts[j]
                    if nxt.get("page_no") != 1:
                        break
                    ntxt = (nxt.get("text") or "").strip()
                    if ntxt and "securities registered pursuant to section 12(g)" not in ntxt.lower():
                        out.append(nxt)
        return out
    except Exception:
        return []
