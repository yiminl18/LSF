def rule_page1_10q_h2_state_and_ein_values(doc: dict) -> list[dict]:
    """Match page-1 10-Q H2 spans that are the state and EIN values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H2":
                if re.fullmatch(r"Delaware|Washington|New York|Jersey", txt, re.I) or re.fullmatch(r"\d{2}-\d{7}", txt):
                    out.append(span)
        return out
    except Exception:
        return []
