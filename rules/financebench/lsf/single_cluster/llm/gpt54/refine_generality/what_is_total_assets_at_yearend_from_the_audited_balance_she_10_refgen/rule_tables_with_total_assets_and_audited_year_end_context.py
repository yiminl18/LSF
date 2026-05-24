def rule_tables_with_total_assets_and_audited_year_end_context(doc: dict) -> list[dict]:
    """Match tables with total assets and year-end wording like 'as of' or fiscal year ended."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "total assets" in txt and ("as of" in txt or "fiscal year ended" in txt or "year ended" in txt):
                out.append(span)
        return out
    except Exception:
        return []
