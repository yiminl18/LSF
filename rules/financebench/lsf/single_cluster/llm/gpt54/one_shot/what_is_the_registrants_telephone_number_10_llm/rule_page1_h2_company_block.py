def rule_page1_h2_company_block(doc: dict) -> list[dict]:
    """Match page-1 H2 company-identification blocks that contain address/EIN/telephone details."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and (span.get("structure", {}) or {}).get("level") == "H2"
            ):
                text = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if any(k in text.lower() for k in [
                    "exact name of registrant",
                    "address of principal executive offices",
                    "registrant’s telephone number",
                    "registrant's telephone number",
                    "zip code"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
