def rule_page1_body_period_line_exact(doc: dict) -> list[dict]:
    """Match body spans on page 1 that are almost exactly the period line."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and ((span.get("structure") or {}).get("level") == "Body"):
                if re.fullmatch(r'For the fiscal year ended .*', txt, re.I) or re.fullmatch(r'For the quarterly period ended .*', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
