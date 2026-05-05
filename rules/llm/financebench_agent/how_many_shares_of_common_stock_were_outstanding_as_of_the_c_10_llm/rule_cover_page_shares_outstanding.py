def rule_cover_page_shares_outstanding(doc: dict) -> list[dict]:
    """Retrieve cover-page body text on pages 1-2 mentioning shares/common stock outstanding."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            if span.get("page_no") not in (1, 2):
                continue
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            hay = f"{path} {txt}".lower()
            if (
                "shares outstanding" in hay
                or "shares of common stock outstanding" in hay
                or "number of shares of common stock outstanding" in hay
                or "common stock issued and outstanding" in hay
                or "common stock outstanding as of" in hay
                or "number of shares outstanding as of" in hay
            ):
                out.append(span)
        return out
    except Exception:
        return []

