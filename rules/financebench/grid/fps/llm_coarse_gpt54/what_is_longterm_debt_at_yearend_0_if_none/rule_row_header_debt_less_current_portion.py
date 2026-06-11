def rule_row_header_debt_less_current_portion(doc: dict) -> list[dict]:
    """Return table spans where a row header contains debt less current portion, a common long-term debt label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in span.get("table_data", {}).get("cells", []):
                txt = c.get("text") or ""
                if c.get("is_row_header") and re.search(r"debt.*less current portion|less current maturities|noncurrent debt", txt, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
