def rule_tables_with_total_assets_and_assets_first(doc: dict) -> list[dict]:
    """Match tables where assets appears before liabilities in the text, as in balance sheets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            a = txt.find("assets")
            l = txt.find("liabilities")
            if a != -1 and l != -1 and a < l:
                out.append(span)
        return out
    except Exception:
        return []
