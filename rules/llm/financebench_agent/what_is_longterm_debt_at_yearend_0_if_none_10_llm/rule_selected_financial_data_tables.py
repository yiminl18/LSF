def rule_selected_financial_data_tables(doc: dict) -> list[dict]:
    """Retrieve selected financial data tables that often include long-term debt year-end values."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            combined = (path + "\n" + text).lower()
            if any(k in combined for k in [
                "selected financial data",
                "selected consolidated financial data",
                "five-year selected",
                "five year selected",
                "selected data"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
