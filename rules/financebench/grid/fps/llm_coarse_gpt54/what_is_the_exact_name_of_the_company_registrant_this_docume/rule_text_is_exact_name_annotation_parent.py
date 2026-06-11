def rule_text_is_exact_name_annotation_parent(doc: dict) -> list[dict]:
    """Match parents of spans whose text is the exact-name-of-registrant annotation."""
    try:
        texts = doc.get("texts", [])
        out = []
        for child in texts:
            txt = (child.get("text", "") or "").strip().lower()
            if "exact name of registrant as specified in its charter" != txt.strip("()"):
                if "exact name of registrant as specified in its charter" not in txt:
                    continue
            pid = child.get("structure", {}).get("parent_id")
            if pid is not None and 0 <= pid < len(texts):
                out.append(texts[pid])
        return out
    except Exception:
        return []
