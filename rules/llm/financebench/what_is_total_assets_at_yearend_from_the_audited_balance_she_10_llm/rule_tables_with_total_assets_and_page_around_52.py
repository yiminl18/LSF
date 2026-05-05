def rule_tables_with_total_assets_and_page_around_52(doc: dict) -> list[dict]:
    """Match total-assets tables around page 52, another common audited statement page."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 50 <= int(span.get("page_no", -1)) <= 54:
                if "total assets" in (span.get("text") or "").lower() or "assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
