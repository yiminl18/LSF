def rule_page1_address_before_securities_registered(doc: dict) -> list[dict]:
    """Match spans just before the 'Securities registered' block on page 1."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "securities registered pursuant to section 12(b)" in txt:
                for j in range(max(0, i - 4), i):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
