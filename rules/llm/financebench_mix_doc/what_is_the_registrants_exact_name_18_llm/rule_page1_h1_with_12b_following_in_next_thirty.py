def rule_page1_h1_with_12b_following_in_next_thirty(doc: dict) -> list[dict]:
    """Match page-1 H1 spans with a Section 12(b) securities-registration block nearby."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            for j in range(i + 1, min(i + 31, len(texts))):
                if texts[j].get("page_no") != 1:
                    break
                t = ((texts[j].get("text", "") or "") + " " + (texts[j].get("text_span", "") or "")).lower()
                if "securities registered pursuant to section 12(b)" in t:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
