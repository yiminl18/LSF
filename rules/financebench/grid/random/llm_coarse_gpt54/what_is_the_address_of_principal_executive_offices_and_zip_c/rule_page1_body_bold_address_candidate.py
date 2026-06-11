def rule_page1_body_bold_address_candidate(doc: dict) -> list[dict]:
    """Match bold body spans on page 1 that look like standalone address fragments."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if ((span.get("structure") or {}).get("level") or "") != "Body":
                continue
            if span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'^\d{2,}|^(?:Santa Monica,|Chicago, IL|Delaware|Washington|New York)$', txt) or re.search(r'\b(?:Plaza|Avenue|Boulevard|Drive|Road|Center)\b', txt):
                out.append(span)
        return out
    except Exception:
        return []
