def rule_page1_h1_h2_h3_period(doc: dict) -> list[dict]:
    """Match page-1 heading spans at levels H1/H2/H3 that contain period language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and level in {"H1", "H2", "H3"}:
                if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
