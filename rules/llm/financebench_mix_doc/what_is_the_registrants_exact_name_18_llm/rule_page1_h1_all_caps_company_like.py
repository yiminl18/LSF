def rule_page1_h1_all_caps_company_like(doc: dict) -> list[dict]:
    """Match page-1 all-caps H1 section headers that look like company names."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and span.get("bold") == 1
                and (span.get("all_cap") == 1 or txt == txt.upper())
                and re.search(r"[A-Z]", txt)
                and "form 10-" not in low
                and "form 8-k" not in low
                and "current report" not in low
                and "securities and exchange commission" not in low
                and "united states" not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
