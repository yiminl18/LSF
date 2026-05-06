def rule_tables_with_december_or_year_end_headers(doc: dict) -> list[dict]:
    """Match tables with year-end/date headers and assets language."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "assets" in text and (
                "december" in text
                or "january" in text
                or "may 31" in text
                or "june 30" in text
                or "august" in text
                or "september" in text
                or "year ended" in text
                or "fiscal year ended" in text
            ):
                out.append(span)
        return out
    except Exception:
        return []
