def rule_page1_cover_block_children(doc: dict) -> list[dict]:
    """Match page-1 child spans under the first company-name cover header block."""
    try:
        texts = doc.get("texts", [])
        company_ids = []
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and "commission" not in txt.lower() and "form 10-k" not in txt.lower():
                    if span.get("structure", {}).get("level") in ("H1", "H2"):
                        company_ids.append(idx)
        if not company_ids:
            return []
        parent_idx = company_ids[0]
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("structure", {}).get("parent_id") == parent_idx:
                out.append(span)
        return out
    except Exception:
        return []
