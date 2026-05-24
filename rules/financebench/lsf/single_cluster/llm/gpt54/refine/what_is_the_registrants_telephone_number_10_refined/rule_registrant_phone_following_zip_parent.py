def rule_registrant_phone_following_zip_parent(doc: dict) -> list[dict]:
    """Match child/body spans under a zip-code-related parent that contain the phone label or number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure", {}) or {}).get("path_text") or "")
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"zip code|\b\d{5}(?:-\d{4})?\b", path, re.I):
                if re.search(r"telephone|area code|registrant|(\(\d{3}\)\s*\d{3}[-\s]?\d{4})|(\+\d{1,3}\s*\d)", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
