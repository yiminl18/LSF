def rule_page1_near_title_of_each_class(doc: dict) -> list[dict]:
    """Match spans near 'Title of each class' on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and "title of each class" in (span.get("text") or "").lower():
                for j in range(max(0, i - 1), min(len(texts), i + 6)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
