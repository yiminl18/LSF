def rule_page1_city_state_zip_only(doc: dict) -> list[dict]:
    """Match page-1 spans that are just ZIP or city/state/ZIP fragments used in split layouts."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"\d{5}(?:-\d{4})?", txt):
                out.append(span)
                continue
            if re.search(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*,?\s+(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+\d{5}(?:-\d{4})?\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
