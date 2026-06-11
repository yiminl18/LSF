def rule_path_text_root_company(doc: dict) -> list[dict]:
    """Match page-1 spans whose path_text equals their own text and whose child mentions exact registrant name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            path = (span.get("structure", {}).get("path_text", "") or "").strip()
            if not txt or txt != path:
                continue
            for child in texts:
                ctext = (child.get("text", "") or "").lower()
                if "exact name of registrant as specified in its charter" in ctext:
                    if child.get("structure", {}).get("parent_id") == i:
                        out.append(span)
                        break
            # also allow text_span annotation on same span
            if "exact name of registrant as specified in its charter" in (span.get("text_span", "") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
