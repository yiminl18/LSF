def rule_profile_of_economy_unemployment_rate_chart_page(doc: dict) -> list[dict]:
    """Match all spans on pages containing an 'Unemployment Rate' heading, for high recall around chart-based layouts."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            if re.search(r"^Unemployment Rate$", span.get("text", "") or "", re.I):
                if isinstance(span.get("page_no"), int):
                    pages.add(span["page_no"])
        for span in doc.get("texts", []):
            if span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
