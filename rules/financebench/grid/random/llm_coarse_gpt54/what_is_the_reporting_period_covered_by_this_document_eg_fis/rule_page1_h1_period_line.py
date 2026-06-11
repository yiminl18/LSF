def rule_page1_h1_period_line(doc: dict) -> list[dict]:
    """Match H1 spans on page 1 that include the period in text_span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and level == "H1":
                if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
