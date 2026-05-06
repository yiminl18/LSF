def rule_page1_page2_only(doc: dict) -> list[dict]:
    """Broad high-recall rule: return all page 1-2 spans mentioning outstanding/common stock/shares."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") in (1, 2):
                t = (span.get("text") or "").lower()
                if "outstanding" in t or "common stock" in t or "shares of common stock" in t:
                    out.append(span)
        return out
    except Exception:
        return []
