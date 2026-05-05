def rule_tables_with_assets_and_no_cash_flows_phrase(doc: dict) -> list[dict]:
    """Match asset tables excluding cash flow statements by phrase."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            low = (span.get("text") or "").lower()
            if "assets" in low and "cash flows" not in low and "cash flow" not in low:
                out.append(span)
        return out
    except Exception:
        return []
