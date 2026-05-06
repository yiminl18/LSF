def rule_page1_near_name_of_each_exchange(doc: dict) -> list[dict]:
    """Match spans near 'Name of each exchange on which registered' on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "name of each exchange on which registered" in txt
                or "name of each exchange" in txt
            ):
                for j in range(max(0, i - 2), min(len(texts), i + 5)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
