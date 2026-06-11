def rule_page1_table_or_text_with_multiple_securities(doc: dict) -> list[dict]:
    """Match page-1 spans that likely contain multiple listed securities, useful for 10-Q/8-K cases with notes and debt listings."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if re.search(r"ordinary shares|notes due|senior notes|common stock", txt, re.I) and re.search(r"trading symbol|exchange|stock exchange|nasdaq", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
