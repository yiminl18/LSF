def rule_tables_with_financial_position_and_total_assets(doc: dict) -> list[dict]:
    """Match statement-of-financial-position tables containing total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "total assets" in low and ("financial position" in low or "financial position" in path):
                out.append(span)
        return out
    except Exception:
        return []
