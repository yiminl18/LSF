def rule_page1_registration_keywords_broad(doc: dict) -> list[dict]:
    """Broad high-recall rule for page 1 spans mentioning registration, symbol, exchange, or listing."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"registered|trading symbol|symbol|exchange|listed on|trades on|section 12\(b\)|name of each exchange", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
