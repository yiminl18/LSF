def rule_page1_company_header_with_embedded_answer(doc: dict) -> list[dict]:
    """Match large page 1 company header spans whose text_span/text embeds the registration block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if re.search(r"securities registered pursuant to section 12\(b\)", txt, re.I) and re.search(r"symbol|exchange|nasdaq|new york stock exchange|nyse", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
