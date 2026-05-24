def rule_partii_sales_revenue_tables_text(doc: dict) -> list[dict]:
    """Retrieve Part II financial/MD&A spans mentioning sales or revenue, favoring substantive pages over TOCs."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label") or ""
            page = span.get("page_no") or 0
            hay = (path + "\n" + text).lower()
            in_partii = "part ii" in hay or "item 7" in hay or "item 8" in hay or "item 6" in hay
            sales_rev = any(k in hay for k in [
                "net sales", "total revenue", "total revenues", "revenue", "revenues",
                "worldwide sales", "analysis of consolidated sales", "sales by segment",
                "statement of earnings", "statement of income", "statement of operations",
                "selected financial data"
            ])
            substantive = page > 3 or any(k in hay for k in [
                "as of and for the year ended", "in millions", "in billions", "dollars in millions", "dollars in billions"
            ])
            if label in {"table", "text", "section_header"} and in_partii and sales_rev and substantive:
                out.append(span)
        return out
    except Exception:
        return []
