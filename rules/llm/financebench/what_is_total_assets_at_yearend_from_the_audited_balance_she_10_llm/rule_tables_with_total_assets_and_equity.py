def rule_tables_with_total_assets_and_equity(doc: dict) -> list[dict]:
    """Match tables containing total assets and equity language."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "total assets" in text and (
                "equity" in text or "shareholders' equity" in text or "stockholders' equity" in text
            ):
                out.append(span)
        return out
    except Exception:
        return []
