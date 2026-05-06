def rule_cover_page_company_identity_keywords(doc: dict) -> list[dict]:
    """Match cover-page spans containing multiple company identity keywords often surrounding the phone."""
    try:
        out = []
        keys = ["exact name of registrant", "state or other jurisdiction", "employer identification", "principal executive offices"]
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1:
                score = sum(1 for k in keys if k in text)
                if score >= 2:
                    out.append(span)
        return out
    except Exception:
        return []
