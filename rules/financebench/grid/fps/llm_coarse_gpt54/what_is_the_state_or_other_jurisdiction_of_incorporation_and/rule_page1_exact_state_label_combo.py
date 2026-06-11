def rule_page1_exact_state_label_combo(doc: dict) -> list[dict]:
    """Match page-1 spans where state value and incorporation label are combined in one span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", txt, re.I) and re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
