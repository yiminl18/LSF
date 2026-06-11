def rule_page1_main_header_with_exchange_inline(doc: dict) -> list[dict]:
    """Match the main company H1 if its inline content includes the exchange answer."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        ex_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market)\b")
        for span in texts:
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                lvl = ((span.get("structure") or {}).get("level") or "")
                if lvl == "H1":
                    low = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
                    if ex_re.search(low):
                        out.append(span)
        return out
    except Exception:
        return []
