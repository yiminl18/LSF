def rule_tables_with_total_assets_and_page_around_42(doc: dict) -> list[dict]:
    """Match total-assets tables around page 42, a common balance-sheet page in many 10-Ks."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 40 <= int(span.get("page_no", -1)) <= 44:
                if "total assets" in (span.get("text") or "").lower() or "assets" in (span.get("text") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
