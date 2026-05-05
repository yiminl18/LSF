def rule_page1_numeric_only_near_number_of_shares_label(doc: dict) -> list[dict]:
    """Match numeric-only spans near 'number of shares of common stock outstanding as of' labels."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            t = (span.get("text") or "")
            if span.get("page_no") not in (1, 2):
                continue
            if re.fullmatch(r"[\d,]{6,}", t.strip()):
                ctx_spans = texts[max(0, i - 4):min(len(texts), i + 1)]
                ctx = " ".join((s.get("text") or "").lower() for s in ctx_spans if s.get("page_no") in (1, 2))
                if "number of shares" in ctx and "outstanding" in ctx:
                    out.append(span)
        return out
    except Exception:
        return []
