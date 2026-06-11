def rule_page1_footlocker_style_company_block(doc: dict) -> list[dict]:
    """Match page-1 company blocks where state, commission file, and EIN are split across adjacent spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                txt = span.get("text", "") or ""
                if "Exact name of registrant as specified in charter" in (span.get("text_span", "") or "") or "Exact name of registrant as specified in its charter" in (span.get("text_span", "") or ""):
                    for cand in texts[i:i+15]:
                        if cand.get("page_no") == 1:
                            out.append(cand)
        return out
    except Exception:
        return []
