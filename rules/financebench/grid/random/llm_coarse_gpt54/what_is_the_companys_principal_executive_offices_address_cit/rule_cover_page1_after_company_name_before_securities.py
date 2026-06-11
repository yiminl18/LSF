def rule_cover_page1_after_company_name_before_securities(doc: dict) -> list[dict]:
    """Match page-1 spans in the company cover block before the securities-registration section starts."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        in_company = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if span.get("label") == "section_header" and txt and txt.upper() == txt and "FORM 10-" not in txt and "SECURITIES AND EXCHANGE" not in txt and len(txt) > 3:
                in_company = True
            if in_company:
                if re.search(r'Securities registered pursuant to Section 12\(b\)', txt, re.I):
                    break
                out.append(span)
        return out
    except Exception:
        return []
