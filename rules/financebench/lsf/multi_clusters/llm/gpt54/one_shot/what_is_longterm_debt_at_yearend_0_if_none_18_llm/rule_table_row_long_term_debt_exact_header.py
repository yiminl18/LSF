def rule_table_row_long_term_debt_exact_header(doc: dict) -> list[dict]:
    """Match tables with a row header exactly equal to long-term debt variants."""
    out = []
    try:
        variants = {
            "long-term debt",
            "long term debt",
            "long-term debt, less current portion",
            "long term debt, less current portion",
        }
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            for c in cells:
                t = (c.get("text") or "").strip().lower()
                if t in variants:
                    out.append(span)
                    break
    except Exception:
        return []
    return out
