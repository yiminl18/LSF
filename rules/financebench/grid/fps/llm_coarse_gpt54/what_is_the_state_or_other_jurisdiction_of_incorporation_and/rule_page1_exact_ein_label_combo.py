def rule_page1_exact_ein_label_combo(doc: dict) -> list[dict]:
    """Match page-1 spans where EIN value and IRS label are combined in one span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r"\d{2}-\d{7}", txt) and re.search(r"(i\.?r\.?s\.?\s+)?employer\s+identification", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
