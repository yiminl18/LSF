def rule_late_page_item7_item8_sections(doc: dict) -> list[dict]:
    """Retrieve later-page Item 7/8 financial sections while excluding front-matter tables of contents."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            page = span.get("page_no") or 0
            hay = f"{path} {text}".lower()
            if page <= 5:
                continue
            if "table of contents" in hay or "index" in hay:
                continue
            if (
                "item 7" in hay
                or "item 8" in hay
                or "management's discussion and analysis" in hay
                or "management’s discussion and analysis" in hay
                or "results of operations" in hay
                or "selected financial data" in hay
                or "financial statements and supplementary data" in hay
                or "financial statements" in hay
                or "statement of income" in hay
                or "statement of earnings" in hay
                or "statement of operations" in hay
                or "net sales" in hay
                or "net earnings" in hay
                or "net income" in hay
                or "net loss" in hay
            ):
                out.append(span)
        return out
    except Exception:
        return []

