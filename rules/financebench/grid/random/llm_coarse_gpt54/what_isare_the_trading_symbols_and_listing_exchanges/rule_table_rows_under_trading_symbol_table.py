def rule_table_rows_under_trading_symbol_table(doc: dict) -> list[dict]:
    """Return table spans that contain data rows under a trading-symbol/exchange table."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            header_texts = [c.get("text", "") or "" for c in cells if c.get("row") == 0]
            all_text = " | ".join(header_texts)
            if re.search(r"trading symbol", all_text, re.I) and re.search(r"exchange", all_text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
