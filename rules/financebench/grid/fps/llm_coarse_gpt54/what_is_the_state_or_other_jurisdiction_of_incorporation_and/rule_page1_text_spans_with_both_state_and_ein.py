def rule_page1_text_spans_with_both_state_and_ein(doc: dict) -> list[dict]:
    """Match page-1 text spans that contain both a state/jurisdiction value and an EIN-like number."""
    try:
        import re
        states = r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)"
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(states, txt, re.I) and re.search(r"\d{2}-\d{7}", txt):
                out.append(span)
        return out
    except Exception:
        return []
