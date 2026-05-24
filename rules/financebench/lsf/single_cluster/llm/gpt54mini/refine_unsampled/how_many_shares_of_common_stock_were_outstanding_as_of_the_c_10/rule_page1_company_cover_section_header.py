def rule_page1_company_cover_section_header(doc: dict) -> list[dict]:
    """Match the main company cover section_header on page 1 when it embeds the answer sentence."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                text = (span.get("text") or "").lower()
                if ("exact name of registrant" in text or "i.r.s. employer identification" in text or "documents incorporated by reference" in text) and ("outstanding" in text or "number of shares of common stock" in text):
                    out.append(span)
        return out
    except Exception:
        return []
