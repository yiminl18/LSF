def rule_page1_bold_address_like_span(doc: dict) -> list[dict]:
    """Match bold page 1 spans that look like address lines with city/state/zip."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "")
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if re.search(r"\b[A-Z][a-zA-Z\.\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", txt):
                out.append(span)
            elif re.search(r"\bUnited Kingdom\b", txt) and re.search(r"\bBristol\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
