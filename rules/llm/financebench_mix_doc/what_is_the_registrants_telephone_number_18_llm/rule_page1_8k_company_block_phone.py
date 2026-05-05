def rule_page1_8k_company_block_phone(doc: dict) -> list[dict]:
    """Match 8-K company identity blocks on page 1 containing the registrant phone line."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"registrant[’'`s]? telephone number, including area code", blob, re.I):
                out.append(span)
            elif span.get("structure", {}).get("level") == "H1" and re.search(r"telephone number", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
