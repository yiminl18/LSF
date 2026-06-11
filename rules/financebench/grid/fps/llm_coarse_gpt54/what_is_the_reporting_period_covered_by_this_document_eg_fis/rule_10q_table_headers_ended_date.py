def rule_10q_table_headers_ended_date(doc: dict) -> list[dict]:
    """Match tables with header cells containing 'ended <month/day>' that can reveal the quarter end date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                txt = (c.get("text") or "").lower()
                if "ended" in txt and re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}', txt):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
