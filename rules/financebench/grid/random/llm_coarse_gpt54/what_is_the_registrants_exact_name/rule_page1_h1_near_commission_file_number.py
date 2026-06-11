def rule_page1_h1_near_commission_file_number(doc: dict) -> list[dict]:
    """Match page-1 H1 headers appearing shortly after a commission file number span."""
    try:
        texts = doc.get("texts", [])
        out = []
        last_commission_idx = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "commission file number" in txt or "commission file no." in txt:
                last_commission_idx = i
            if (
                last_commission_idx is not None
                and 0 < i - last_commission_idx <= 8
                and span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text") or "").upper()
            ):
                out.append(span)
        return out
    except Exception:
        return []
