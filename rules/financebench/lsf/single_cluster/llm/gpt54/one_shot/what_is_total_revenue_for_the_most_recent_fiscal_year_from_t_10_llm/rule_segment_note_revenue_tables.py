def rule_segment_note_revenue_tables(doc: dict) -> list[dict]:
    """Match tables in segment/geographic note contexts that may also contain total revenue for the latest year."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            txt = (span.get("text") or "").lower()
            if any(k in path + " " + txt for k in ["segment information", "segments of business", "geographic areas"]):
                if any(k in txt for k in ["revenue", "revenues", "sales", "net sales", "net revenues"]):
                    out.append(span)
        return out
    except Exception:
        return []
