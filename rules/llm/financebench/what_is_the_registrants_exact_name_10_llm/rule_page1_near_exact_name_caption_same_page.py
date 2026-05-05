def rule_page1_near_exact_name_caption_same_page(doc: dict) -> list[dict]:
    """Match spans within a small page-1 neighborhood around the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if "exact name of registrant" in ((span.get("text") or "").lower()):
                for j in range(max(0, i - 2), min(len(texts), i + 2)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
