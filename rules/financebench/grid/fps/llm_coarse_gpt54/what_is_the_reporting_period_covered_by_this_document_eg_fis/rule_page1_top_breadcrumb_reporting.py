def rule_page1_top_breadcrumb_reporting(doc: dict) -> list[dict]:
    """Match top-of-document spans with shallow depth and reporting-period keywords."""
    try:
        out = []
        for span in doc.get("texts", []):
            struct = span.get("structure") or {}
            if span.get("page_no") == 1 and (struct.get("depth") or 99) <= 3:
                text = (span.get("text") or "").lower()
                if any(k in text for k in [
                    "fiscal year ended", "quarterly period ended", "date of report", "for the period ending"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
