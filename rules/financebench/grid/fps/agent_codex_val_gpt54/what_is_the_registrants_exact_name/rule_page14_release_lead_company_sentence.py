def rule_page14_release_lead_company_sentence(doc: dict) -> list[dict]:
    """Match early release lead sentences that name the company before a ticker marker."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (span.get("page_no") or 999) > 4:
                continue
            if span.get("label") != "text":
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if len(text) > 800:
                continue
            if not any(phrase in low for phrase in ("today reported", "today announced", "today released")):
                continue
            if "[" not in text and "(nasdaq:" not in low and "(nyse:" not in low and " nasdaq:" not in low and " nyse:" not in low:
                continue
            out.append(span)
        return out
    except Exception:
        return []
