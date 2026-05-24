def rule_page1_h1_company_block(doc: dict) -> list[dict]:
    """Match large company-identification header blocks on page 1 that often embed the telephone number."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and (span.get("structure", {}) or {}).get("level") == "H1"
            ):
                text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if any(k in text.lower() for k in [
                    "exact name of registrant",
                    "address of principal executive offices",
                    "registrant’s telephone number",
                    "registrant's telephone number",
                    "address and telephone number"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
