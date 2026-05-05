def rule_tables_with_assets_and_noncurrent_assets(doc: dict) -> list[dict]:
    """Match tables containing current assets and noncurrent/other assets patterns."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "current assets" in low and ("other assets" in low or "noncurrent assets" in low or "non-current assets" in low):
                out.append(span)
        return out
    except Exception:
        return []
