def rule_page1_ein_header_then_body_label(doc: dict) -> list[dict]:
    """Match page-1 EIN-value section headers whose child/body span is the IRS identification label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if re.fullmatch(r"\d{2}-\d{7}", txt):
                    for child in texts:
                        if child.get("page_no") == 1 and child.get("structure", {}).get("parent_id") == idx:
                            ctext = (child.get("text") or "").strip()
                            if re.search(r"i\.r\.s\. employer identification no\.?|irs employer identification no\.?|employer identification no", ctext, re.I):
                                out.append(span)
                                out.append(child)
        return out
    except Exception:
        return []
