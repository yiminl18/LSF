def rule_page1_large_bold_company_header(doc: dict) -> list[dict]:
    """Match large bold page-1 headers that are likely the registrant name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        bad = re.compile(r"(FORM 10-|FORM 8-K|CURRENT REPORT|SECURITIES AND EXCHANGE COMMISSION|TABLE OF CONTENTS|INDEX)", re.I)
        for span in texts:
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and (span.get("size") or 0) >= 10
                and txt
                and not bad.search(txt)
                and span.get("label") in {"section_header", "text"}
            ):
                if len(txt.split()) >= 2:
                    out.append(span)
        return out
    except Exception:
        return []
