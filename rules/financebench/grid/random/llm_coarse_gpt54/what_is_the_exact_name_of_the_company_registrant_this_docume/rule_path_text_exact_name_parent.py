def rule_path_text_exact_name_parent(doc: dict) -> list[dict]:
    """Match spans whose path_text is the company name and that have a child text saying exact name of registrant."""
    try:
        texts = doc.get("texts", [])
        parent_ids = set()
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "exact name of registrant" in txt:
                pid = span.get("structure", {}).get("parent_id")
                if pid is not None:
                    parent_ids.add(pid)
        out = []
        for pid in parent_ids:
            if 0 <= pid < len(texts):
                out.append(texts[pid])
        return out
    except Exception:
        return []
