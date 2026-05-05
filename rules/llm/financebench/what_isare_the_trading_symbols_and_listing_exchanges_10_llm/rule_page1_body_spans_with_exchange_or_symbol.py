def rule_page1_body_spans_with_exchange_or_symbol(doc: dict) -> list[dict]:
    """Match page 1 body spans containing either exchange names or symbol cues."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if ((span.get("structure", {}) or {}).get("level") or "") != "Body":
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if any(k in txt for k in [
                "trading symbol",
                "under the symbol",
                "nasdaq",
                "new york stock exchange",
                "nyse",
                "exchange on which registered",
            ]):
                out.append(span)
        return out
    except Exception:
        return []
