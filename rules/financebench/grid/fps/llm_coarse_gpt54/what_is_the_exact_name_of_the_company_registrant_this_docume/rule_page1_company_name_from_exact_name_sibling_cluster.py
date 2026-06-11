def rule_page1_company_name_from_exact_name_sibling_cluster(doc: dict) -> list[dict]:
    """Match spans in the same sibling cluster as the exact-name annotation, preferring the first bold/large one."""
    try:
        texts = doc.get("texts", [])
        out = []
        for child in texts:
            if child.get("page_no") != 1:
                continue
            if "exact name of registrant as specified in its charter" not in (child.get("text", "") or "").lower():
                continue
            pid = child.get("structure", {}).get("parent_id")
            if pid is not None and 0 <= pid < len(texts):
                out.append(texts[pid])
                continue
            # fallback: previous span
            idx = texts.index(child)
            if idx > 0:
                out.append(texts[idx - 1])
        return out
    except Exception:
        return []
