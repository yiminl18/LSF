def rule_news_release_any_page1(doc: dict) -> list[dict]:
    """Match any page-1 span explicitly saying News Release."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and (span.get("text") or "").strip().lower() == "news release":
                out.append(span)
        return out
    except Exception:
        return []
