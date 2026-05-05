def rule_tables_with_total_assets_and_shareholders_equity(doc: dict) -> list[dict]:
    """Match balance-sheet-like tables containing total assets and shareholders/stockholders equity."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "total assets" in low and (
                "shareholders' equity" in low
                or "stockholders' equity" in low
                or "shareowners' equity" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
