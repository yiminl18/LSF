def rule_page1_h1_company_header_block(doc: dict) -> list[dict]:
    """Match large company-name header blocks on page 1 that often embed the phone number in text_span."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and (span.get("size") or 0) >= 10
            ):
                text = span.get("text", "") or ""
                text_span = span.get("text_span", "") or ""
                blob = text + " " + text_span
                if "registrant" in blob.lower() or "telephone number" in blob.lower() or "principal executive offices" in blob.lower():
                    out.append(span)
        return out
    except Exception:
        return []
