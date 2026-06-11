def rule_page1_all_spans_near_symbol_like(doc: dict) -> list[dict]:
    """Match a small neighborhood around page-1 ticker-like spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            txt = (s.get("text") or "").strip()
            if s.get("page_no") == 1 and re.fullmatch(r"[A-Z]{1,6}(?:\d+[A-Z]*)?(?:[./-][A-Z0-9]+)?", txt):
                for j in range(max(0, i - 3), min(len(texts), i + 4)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
