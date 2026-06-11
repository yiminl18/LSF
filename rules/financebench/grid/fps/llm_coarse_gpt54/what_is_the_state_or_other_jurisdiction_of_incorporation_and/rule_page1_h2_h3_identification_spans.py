def rule_page1_h2_h3_identification_spans(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 identification spans that commonly hold state or EIN values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = (span.get("structure", {}) or {}).get("level")
            if level not in {"H2", "H3"}:
                continue
            txt = (span.get("text") or "").strip()
            tspan = (span.get("text_span") or "").strip()
            if re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
            elif re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", txt, re.I):
                out.append(span)
            elif re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", tspan, re.I):
                out.append(span)
            elif re.search(r"employer\s+identification", tspan, re.I):
                out.append(span)
        return out
    except Exception:
        return []
