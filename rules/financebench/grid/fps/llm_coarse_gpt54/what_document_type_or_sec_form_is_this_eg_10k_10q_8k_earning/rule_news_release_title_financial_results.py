def rule_news_release_title_financial_results(doc: dict) -> list[dict]:
    """Match titles under News Release that announce financial results."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").lower()
            text = (span.get("text") or "").strip()
            if "news release" in path and re.search(r"financial results", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
