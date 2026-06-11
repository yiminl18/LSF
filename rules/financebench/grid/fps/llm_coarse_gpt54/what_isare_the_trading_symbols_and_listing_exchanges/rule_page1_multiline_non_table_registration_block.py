def rule_page1_multiline_non_table_registration_block(doc: dict) -> list[dict]:
    """Match non-table page-1 registration blocks where class, symbol, and exchange are split across consecutive text spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 5):
            window = texts[i:i + 8]
            if not all(w.get("page_no") == 1 for w in window):
                continue
            joined = " ".join((w.get("text") or "") for w in window)
            if re.search(r"Title of each class", joined, re.I) and re.search(r"Trading Symbol", joined, re.I) and re.search(r"exchange on which registered", joined, re.I):
                out.extend(window)
        return out
    except Exception:
        return []
