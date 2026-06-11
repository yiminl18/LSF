def rule_page1_near_exact_name_caption_any_direction(doc: dict) -> list[dict]:
    """Match spans within a small window around any exact-name caption on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if "exact name of registrant" in ((span.get("text") or "").lower()):
                for j in range(max(0, i - 2), min(len(texts), i + 3)):
                    if j != i:
                        out.append(texts[j])
        return out
    except Exception:
        return []
