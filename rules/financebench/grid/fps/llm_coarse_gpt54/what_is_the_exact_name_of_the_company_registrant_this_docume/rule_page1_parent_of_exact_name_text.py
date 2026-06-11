def rule_page1_parent_of_exact_name_text(doc: dict) -> list[dict]:
    """Match any span on page 1 that is the parent of a body span containing the exact-name annotation."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen = set()
        for i, child in enumerate(texts):
            if child.get("page_no") != 1:
                continue
            txt = (child.get("text", "") or "").lower()
            if "exact name of registrant as specified in its charter" in txt:
                pid = child.get("structure", {}).get("parent_id")
                if pid is not None and pid not in seen and 0 <= pid < len(texts):
                    out.append(texts[pid])
                    seen.add(pid)
        return out
    except Exception:
        return []
