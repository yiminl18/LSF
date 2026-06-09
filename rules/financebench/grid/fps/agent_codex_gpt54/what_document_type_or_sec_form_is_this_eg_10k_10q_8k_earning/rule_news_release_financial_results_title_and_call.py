def rule_news_release_financial_results_title_and_call(doc: dict) -> list[dict]:
    """Match news-release filings via the bold financial-results title and earnings-results call line."""
    try:
        results = []
        for s in doc.get("texts", []):
            text = " ".join((s.get("text") or "").split())
            lowered = text.lower()
            if s.get("label") not in {"text", "section_header"}:
                continue
            if (
                (s.get("page_no") or 99) <= 2
                and s.get("bold") == 1
                and "financial results" in lowered
            ):
                results.append(s)
                continue
            if "earnings results conference call" in lowered:
                results.append(s)
        return results
    except Exception:
        return []
