def rule_exact_name_caption_parent(doc: dict) -> list[dict]:
    """Match spans whose text is the parent heading of a nearby '(Exact name of registrant...)' caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().lower()
            if "exact name of registrant" in txt or "exact name of registrant as specified in its charter" in txt:
                parent_id = span.get("structure", {}).get("parent_id")
                if parent_id is not None and 0 <= parent_id < len(texts):
                    out.append(texts[parent_id])
                if i > 0:
                    prev = texts[i - 1]
                    if prev not in out and prev.get("page_no") == span.get("page_no"):
                        out.append(prev)
        return out
    except Exception:
        return []
