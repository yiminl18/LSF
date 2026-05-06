def rule_balance_sheet_tables(doc: dict) -> list[dict]:
    """Retrieve tables that are balance sheets or statements of financial position."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            combined = (path + "\n" + text).lower()
            if any(k in combined for k in [
                "consolidated balance sheets",
                "consolidated balance sheet",
                "balance sheets",
                "balance sheet",
                "statement of financial position",
                "statements of financial position",
                "financial position"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
