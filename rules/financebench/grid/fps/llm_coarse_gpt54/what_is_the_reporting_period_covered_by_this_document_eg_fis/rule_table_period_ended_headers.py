def rule_table_period_ended_headers(doc: dict) -> list[dict]:
    """Match tables whose header cells include 'period ended' or 'months ended' dates."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                txt = (c.get("text") or "").lower()
                if c.get("is_column_header") and (
                    "period ended" in txt or "months ended" in txt or "year ended" in txt
                ):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
