def rule_page1_cover_block_children_with_ein_or_state(doc: dict) -> list[dict]:
    """Match page-1 child spans under the company cover block that are likely state or EIN values/labels."""
    try:
        import re
        texts = doc.get("texts", [])
        company_ids = []
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and "commission" not in txt.lower() and "form 10-k" not in txt.lower():
                    company_ids.append(idx)
        if not company_ids:
            return []
        parent_idx = company_ids[0]
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("structure", {}).get("parent_id") == parent_idx:
                txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if re.search(r"\d{2}-\d{7}|incorporation|jurisdiction|employer identification|state of incorporation", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
