def rule_page1_first_10_spans_form_signals(doc: dict) -> list[dict]:
    """Match strong document-type signals within the first 10 spans on page 1."""
    import re
    try:
        out = []
        count = 0
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            count += 1
            if count > 10:
                break
            text = (span.get("text") or "").strip()
            if re.search(r"\b(FORM 10-K|FORM 10-Q|FORM 8-K|CURRENT REPORT|NEWS RELEASE)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
