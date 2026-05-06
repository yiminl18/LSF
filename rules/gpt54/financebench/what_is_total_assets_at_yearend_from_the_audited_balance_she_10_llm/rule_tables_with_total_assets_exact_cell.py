def rule_tables_with_total_assets_exact_cell(doc: dict) -> list[dict]:
    """Match tables containing an exact cell text of total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in (span.get("table_data") or {}).get("cells", []):
                if (c.get("text") or "").strip().lower() == "total assets":
                    out.append(span)
                    break
        return out
    except Exception:
        return []
