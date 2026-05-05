def rule_parent_of_page1_exact_name_caption(doc: dict) -> list[dict]:
    """Match the parent section_header of the exact-name caption on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if "exact name of registrant" not in txt:
                continue
            parent_id = span.get("structure", {}).get("parent_id")
            if parent_id is not None and 0 <= parent_id < len(texts):
                parent = texts[parent_id]
                if parent.get("page_no") == 1:
                    out.append(parent)
        return out
    except Exception:
        return []
