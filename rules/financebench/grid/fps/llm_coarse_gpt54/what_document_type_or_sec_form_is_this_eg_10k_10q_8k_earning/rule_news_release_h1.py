def rule_news_release_h1(doc: dict) -> list[dict]:
    """Match top-level News Release headers, useful for earnings release documents."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip().lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and str(span.get("structure", {}).get("level", "")) == "H1"
                and "news release" == text
            ):
                out.append(span)
        return out
    except Exception:
        return []
