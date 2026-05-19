def rule_net_income_row_in_financial_statements_section(doc: dict) -> list[dict]:
    """Match tables in Item 8 / Financial Statements and Supplementary Data with a net income row."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if not re.search(r"(item\s*8|financial statements and supplementary data)", path, re.I):
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for vals in row_text.values():
                joined = " | ".join(vals)
                if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", joined, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
