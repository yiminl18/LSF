def rule_8k_cover_current_report_context(doc: dict) -> list[dict]:
    """Match early Form 8-K cover spans that establish the filing is a current report with no year-end debt table."""
    try:
        hits: list[dict] = []
        for span in doc.get("texts", []):
            page_no = span.get("page_no") or 99
            if page_no > 2:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            text = " ".join((span.get("text") or "").split()).lower()
            if len(text) > 140:
                continue
            if (
                "form 8-k" in text
                or "current report" in text
                or text.startswith("item 8.01")
                or text.startswith("item 5.07")
                or text.startswith("item 9.01")
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
