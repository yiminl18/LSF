def rule_same_parent_as_exact_name_caption(doc: dict) -> list[dict]:
    """Match sibling spans sharing the same parent as the exact-name caption, favoring the first sibling."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if "exact name of registrant" not in ((span.get("text") or "").lower()):
                continue
            parent_id = span.get("structure", {}).get("parent_id")
            if parent_id is None:
                continue
            siblings = [s for s in texts if s.get("structure", {}).get("parent_id") == parent_id]
            siblings = [s for s in siblings if "exact name of registrant" not in ((s.get("text") or "").lower())]
            out.extend(siblings[:2])
        return out
    except Exception:
        return []
