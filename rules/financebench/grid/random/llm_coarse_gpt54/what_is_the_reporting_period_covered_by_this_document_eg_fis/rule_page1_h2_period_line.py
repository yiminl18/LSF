def rule_page1_h2_period_line(doc: dict) -> list[dict]:
    """Match H2 spans on page 1 that are period lines, a common pattern in Costco/Amcor/Adobe 10-Q."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and level == "H2":
                if re.search(r'(For the fiscal year ended|For the quarterly period ended|Date of Report)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
