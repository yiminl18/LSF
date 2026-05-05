def rule_selected_financial_data_tables(doc: dict) -> list[dict]:
    """Retrieve selected financial data tables and adjacent headers in Part II."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label") or ""
            page = span.get("page_no") or 0
            hay = (path + "\n" + text).lower()
            if page >= 4 and label in {"table", "text", "section_header"}:
                if (
                    "selected financial data" in hay
                    or "five-year consolidated" in hay
                    or "years ended december 31" in hay
                    or "years ended may 31" in hay
                    or "as of and for the year ended" in hay
                ):
                    out.append(span)
        return out
    except Exception:
        return []
