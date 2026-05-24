def rule_parent_of_exact_name_caption(doc: dict) -> list[dict]:
    """Match the parent span of any caption '(Exact name of registrant as specified in its charter)'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if "exact name of registrant" in (span.get("text", "") or "").lower():
                parent_id = span.get("structure", {}).get("parent_id")
                if parent_id is None:
                    continue
                if 0 <= parent_id < len(texts):
                    out.append(texts[parent_id])
        return out
    except Exception:
        return []
