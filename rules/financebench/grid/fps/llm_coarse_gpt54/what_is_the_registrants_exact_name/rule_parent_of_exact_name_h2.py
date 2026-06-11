def rule_parent_of_exact_name_h2(doc: dict) -> list[dict]:
    """Match H1/H2/H3 parent spans of exact-name caption spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if "exact name of registrant" not in ((span.get("text") or "").lower()):
                continue
            parent_id = span.get("structure", {}).get("parent_id")
            if parent_id is None or not (0 <= parent_id < len(texts)):
                continue
            parent = texts[parent_id]
            if parent.get("structure", {}).get("level") in {"H1", "H2", "H3"} or parent.get("label") == "section_header":
                out.append(parent)
        return out
    except Exception:
        return []
