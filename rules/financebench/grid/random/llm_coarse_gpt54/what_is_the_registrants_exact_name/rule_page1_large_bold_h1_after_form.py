def rule_page1_large_bold_h1_after_form(doc: dict) -> list[dict]:
    """Match large bold H1 headers on page 1 that appear after a FORM 10-K/10-Q/8-K header."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_form = False
        for span in texts:
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and "FORM 10-" in txt.upper():
                seen_form = True
            if (
                seen_form
                and span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and span.get("bold") == 1
                and float(span.get("size") or 0) >= 10
            ):
                if "FORM 10-" not in txt.upper() and "SECURITIES AND EXCHANGE COMMISSION" not in txt.upper():
                    out.append(span)
        return out
    except Exception:
        return []
