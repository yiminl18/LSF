def rule_page1_main_identity_block(doc: dict) -> list[dict]:
    """Match the main page-1 identity block containing company name, address, and phone."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"exact name of registrant|principal executive offices|telephone number|area code", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
