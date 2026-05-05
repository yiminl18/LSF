def rule_page1_first_ein_and_neighbors(doc: dict) -> list[dict]:
    """Match the first EIN-like span on page 1 plus nearby neighbors."""
    import re
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and re.search(r"\b\d{2}-\d{7}\b", span.get("text", "") or ""):
                out = []
                for j in range(max(0, i - 4), min(len(texts), i + 4)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
                return out
        return []
    except Exception:
        return []
