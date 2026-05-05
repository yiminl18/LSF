def rule_substantive_partii_sales_revenue_text(doc: dict) -> list[dict]:
    """Retrieve substantive Part II text spans discussing sales or revenue totals."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label") or ""
            page = span.get("page_no") or 0
            hay = (path + "\n" + text).lower()
            if label not in {"text", "section_header", "table"}:
                continue
            if page < 4:
                continue
            if not ("part ii" in hay or "item 7" in hay or "item 8" in hay or "item 6" in hay):
                continue
            if any(k in hay for k in [
                "net sales for the year ended",
                "worldwide sales increased",
                "worldwide sales",
                "sales by segment",
                "analysis of consolidated sales",
                "total revenue",
                "total revenues",
                "net sales",
                "revenue for the year ended",
                "revenues for the year ended"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
