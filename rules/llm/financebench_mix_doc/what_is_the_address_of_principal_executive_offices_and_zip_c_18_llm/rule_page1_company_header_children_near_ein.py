def rule_page1_company_header_children_near_ein(doc: dict) -> list[dict]:
    """Match page-1 child spans under the company H1 that occur near EIN/address block."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        company_ids = set()
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                t = (span.get("text") or "").strip()
                if t and re.search(r'(inc\.|corporation|company|plc|com, inc\.|blizzard)', t, re.I):
                    company_ids.add(i)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if re.search(r'(inc\.|corporation|company|plc|com, inc\.|blizzard)', path, re.I):
                txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r'address|zip code|employer identification|telephone number', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
