def rule_page1_3m_style_h2_values(doc: dict) -> list[dict]:
    """Match page-1 H2 section headers that are state or EIN values in split-value cover pages."""
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
