def rule_exchange_in_any_page1_table_or_neighbor(doc: dict) -> list[dict]:
    """Match page-1 tables with exchange info and nearby spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'new york stock exchange|nasdaq|global select market', txt, re.I):
                    out.append(span)
                    for j in range(max(0, i - 2), min(len(texts), i + 3)):
                        if texts[j].get("page_no") == 1:
                            out.append(texts[j])
        return out
    except Exception:
        return []
