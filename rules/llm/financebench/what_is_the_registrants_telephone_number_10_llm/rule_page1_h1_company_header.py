def rule_page1_h1_company_header(doc: dict) -> list[dict]:
    """Match the main company H1/header block on page 1, which often embeds the phone number."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                level = ((span.get("structure") or {}).get("level") or "")
                if level == "H1":
                    text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                    if "exact name of registrant" in text.lower():
                        out.append(span)
        return out
    except Exception:
        return []
