def rule_page1_identification_path_depth_2_3(doc: dict) -> list[dict]:
    """Match page-1 depth 2-3 spans in the company block with state/EIN patterns."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            depth = (span.get("structure", {}) or {}).get("depth")
            if depth not in {2, 3}:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", txt, re.I):
                out.append(span)
            elif re.search(r"employer\s+identification", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"\d{2}-\d{7}", (span.get("text") or "").strip()):
                out.append(span)
            elif re.fullmatch(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", (span.get("text") or "").strip(), re.I):
                out.append(span)
        return out
    except Exception:
        return []
