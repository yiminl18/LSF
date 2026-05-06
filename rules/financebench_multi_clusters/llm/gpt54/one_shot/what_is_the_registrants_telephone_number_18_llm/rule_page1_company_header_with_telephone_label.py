def rule_page1_company_header_with_telephone_label(doc: dict) -> list[dict]:
    """Match company H1 headers whose text_span includes a telephone label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1":
                text = span.get("text", "") or ""
                if re.search(r"form 10-|current report|commission", text, re.I):
                    continue
                ts = span.get("text_span", "") or ""
                if re.search(r"telephone number|area code|principal executive offices", ts, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
