def rule_page1_h1_company_block(doc: dict) -> list[dict]:
    """Match the main company H1 block on page 1, which often contains the address in text_span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                txt = span.get("text") or ""
                tsp = span.get("text_span") or ""
                if "exact name of registrant" in tsp.lower() or "exact name of registrant" in txt.lower():
                    out.append(span)
        return out
    except Exception:
        return []
