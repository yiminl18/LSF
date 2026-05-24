def rule_financial_statement_tables_long_term_debt(doc: dict) -> list[dict]:
    """Retrieve financial statement tables likely containing long-term debt year-end values."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            combined = (path + "\n" + text).lower()
            if not any(k in combined for k in [
                "item 8", "financial statements", "financial statement", "supplementary data",
                "consolidated balance", "balance sheet", "statement of financial position",
                "financial position", "long-term debt", "long term debt", "long-term borrowings",
                "long term borrowings", "debt"
            ]):
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            cell_text = " ".join((c.get("text") or "") for c in cells).lower()
            if any(k in cell_text for k in [
                "long-term debt", "long term debt", "long-term borrowings", "long term borrowings",
                "total debt", "borrowings", "debt"
            ]) or any(k in combined for k in ["balance sheet", "statement of financial position", "consolidated balance"]):
                out.append(span)
        return out
    except Exception:
        return []
