def rule_page1_10q_8k_h2_phone_block(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 cover blocks in 10-Q/8-K filings that contain the phone and label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            lvl = span.get("structure", {}).get("level")
            if span.get("page_no") == 1 and lvl in {"H2", "H3"}:
                blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
                if re.search(r"registrant[’'`s]? telephone number|telephone number, including area code", blob, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
