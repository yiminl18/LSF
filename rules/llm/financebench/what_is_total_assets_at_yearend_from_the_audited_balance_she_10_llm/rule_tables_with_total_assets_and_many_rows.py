def rule_tables_with_total_assets_and_many_rows(doc: dict) -> list[dict]:
    """Match larger financial statement tables containing total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            num_rows = ((span.get("table_data") or {}).get("num_rows") or 0)
            txt = (span.get("text") or "").lower()
            if "total assets" in txt and num_rows >= 8:
                out.append(span)
        return out
    except Exception:
        return []
