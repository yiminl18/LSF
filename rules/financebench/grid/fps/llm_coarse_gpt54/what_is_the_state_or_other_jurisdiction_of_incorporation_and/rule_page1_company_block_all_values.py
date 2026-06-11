def rule_page1_company_block_all_values(doc: dict) -> list[dict]:
    """Match all page-1 spans in the company identification block containing state, file number, EIN, address labels."""
    try:
        import re
        out = []
        started = False
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if not started and re.search(r"exact name of registrant", txt, re.I):
                started = True
            if started:
                if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", txt, re.I):
                    out.append(span)
                elif re.search(r"commission file", txt, re.I):
                    out.append(span)
                elif re.search(r"employer\s+identification", txt, re.I):
                    out.append(span)
                elif re.fullmatch(r"\d{2}-\d{7}", (span.get("text") or "").strip()):
                    out.append(span)
                elif re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", (span.get("text") or "").strip(), re.I):
                    out.append(span)
        return out
    except Exception:
        return []
