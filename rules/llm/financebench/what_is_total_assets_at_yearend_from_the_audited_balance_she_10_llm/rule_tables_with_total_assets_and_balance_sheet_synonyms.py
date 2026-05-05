def rule_tables_with_total_assets_and_balance_sheet_synonyms(doc: dict) -> list[dict]:
    """Match tables using common balance-sheet synonyms plus total assets."""
    try:
        out = []
        syns = [
            "balance sheet",
            "balance sheets",
            "statement of financial position",
            "financial position",
        ]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "total assets" in txt and any(s in txt or s in path for s in syns):
                out.append(span)
        return out
    except Exception:
        return []
