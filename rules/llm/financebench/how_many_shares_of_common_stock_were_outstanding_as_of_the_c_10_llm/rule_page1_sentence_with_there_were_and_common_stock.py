def rule_page1_sentence_with_there_were_and_common_stock(doc: dict) -> list[dict]:
    """Match page-1 spans using 'There were ... common stock ... outstanding' style wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "there were" in t and "common stock" in t and "outstanding" in t:
                out.append(span)
    except Exception:
        return []
    return out
