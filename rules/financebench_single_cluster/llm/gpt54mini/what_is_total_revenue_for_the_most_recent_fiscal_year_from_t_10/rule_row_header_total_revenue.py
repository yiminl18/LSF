def rule_row_header_total_revenue(doc: dict) -> list[dict]:
    """Match tables containing a row header exactly or nearly equal to total revenue / total revenues."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                txt = (c.get("text") or "").strip().lower()
                if txt in {"total revenue", "total revenues"}:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
