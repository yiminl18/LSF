def rule_tables_with_total_assets_and_usd_units(doc: dict) -> list[dict]:
    """Match total-assets tables with U.S. dollar unit wording."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "total assets" in txt and ("u.s. dollars" in txt or "dollars" in txt or "$" in txt):
                out.append(span)
        return out
    except Exception:
        return []
