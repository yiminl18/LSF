def rule_page1_first_h1_after_commission_file(doc: dict) -> list[dict]:
    """Match the first page-1 H1 after the last commission file number mention on the cover page."""
    try:
        texts = doc.get("texts", [])
        last_idx = -1
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and ("commission file number" in txt or "commission file no." in txt):
                last_idx = i
        if last_idx >= 0:
            for span in texts[last_idx + 1:]:
                if (
                    span.get("page_no") == 1
                    and span.get("label") == "section_header"
                    and span.get("structure", {}).get("level") == "H1"
                    and "FORM 10-" not in (span.get("text") or "").upper()
                ):
                    return [span]
        return []
    except Exception:
        return []
