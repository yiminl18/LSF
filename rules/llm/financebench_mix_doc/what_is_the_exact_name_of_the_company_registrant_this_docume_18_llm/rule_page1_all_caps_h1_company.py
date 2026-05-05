def rule_page1_all_caps_h1_company(doc: dict) -> list[dict]:
    """Match all-caps H1 company-name headers on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        bad = re.compile(r"(FORM 10-|FORM 8-K|CURRENT REPORT|SECURITIES AND EXCHANGE COMMISSION|UNITED STATES)", re.I)
        for span in texts:
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and span.get("label") == "section_header"
                and txt
                and txt.upper() == txt
                and not bad.search(txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
