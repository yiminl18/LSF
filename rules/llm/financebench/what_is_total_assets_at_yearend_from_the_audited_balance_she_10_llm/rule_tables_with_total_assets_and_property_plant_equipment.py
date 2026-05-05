def rule_tables_with_total_assets_and_property_plant_equipment(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and PP&E language."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and (
                "property, plant and equipment" in low
                or "property and equipment" in low
                or "property plant and equipment" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
