def rule_page1_numeric_span_after_common_stock_label(doc: dict) -> list[dict]:
    """Match numeric spans following a nearby 'common stock outstanding' label within 5 spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") not in (1, 2):
                continue
            if not re.fullmatch(r"[\d,]{6,}", (span.get("text") or "").strip()):
                continue
            ctx = " ".join((texts[j].get("text") or "").lower() for j in range(max(0, i - 5), i) if texts[j].get("page_no") in (1, 2))
            if "common stock" in ctx and "outstanding" in ctx:
                out.append(span)
        return out
    except Exception:
        return []
