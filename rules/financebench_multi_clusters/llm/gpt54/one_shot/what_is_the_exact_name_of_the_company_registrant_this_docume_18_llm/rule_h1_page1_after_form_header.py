def rule_h1_page1_after_form_header(doc: dict) -> list[dict]:
    """Match H1/section_header spans on page 1 appearing after a FORM 10-K/10-Q/8-K header and before later content."""
    try:
        texts = doc.get("texts", [])
        out = []
        form_seen = False
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if "FORM 10-" in txt or txt == "FORM 8-K":
                form_seen = True
                continue
            if form_seen and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                if "PART I" not in txt and "CURRENT REPORT" not in txt and "SECURITIES AND EXCHANGE COMMISSION" not in txt:
                    out.append(span)
        return out
    except Exception:
        return []
