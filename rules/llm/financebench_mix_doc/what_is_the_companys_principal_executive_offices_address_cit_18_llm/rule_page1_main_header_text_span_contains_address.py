def rule_page1_main_header_text_span_contains_address(doc: dict) -> list[dict]:
    """Match company H1 spans whose text_span contains the address and principal office label."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                tsp = (span.get("text_span") or "").lower()
                if "address of principal executive offices" in tsp or "address and telephone number, including area code, of registrant’s principal executive offices" in tsp or "address and telephone number, including area code, of registrant's principal executive offices" in tsp:
                    out.append(span)
        return out
    except Exception:
        return []
