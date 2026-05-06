def rule_page1_text_with_city_state_zip_and_principal_label(doc: dict) -> list[dict]:
    """Match page 1 text spans containing city/state/zip plus principal office label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = span.get("text") or ""
            low = txt.lower()
            if "principal executive offices" in low:
                out.append(span)
            elif re.search(r"\b[A-Z][a-zA-Z\.\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+\d{5}", txt):
                out.append(span)
        return out
    except Exception:
        return []
