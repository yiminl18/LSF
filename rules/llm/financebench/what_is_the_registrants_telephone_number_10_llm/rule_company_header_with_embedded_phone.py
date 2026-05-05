def rule_company_header_with_embedded_phone(doc: dict) -> list[dict]:
    """Match the main company header block when it embeds address/EIN/telephone details."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"exact name of registrant", text, re.I) and re.search(r"(\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\+\d{1,3}\s*\d)", text):
                out.append(span)
        return out
    except Exception:
        return []
