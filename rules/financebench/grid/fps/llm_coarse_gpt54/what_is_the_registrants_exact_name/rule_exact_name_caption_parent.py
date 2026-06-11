def rule_exact_name_caption_parent(doc: dict) -> list[dict]:
    """Match spans whose text is the parent heading of a nearby '(Exact name of registrant...)' caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().lower()
            if "exact name of registrant" in txt:
                parent_id = span.get("structure", {}).get("parent_id")
                if parent_id is not None and 0 <= parent_id < len(texts):
                    out.append(texts[parent_id])
        return out
    except Exception:
        return []
