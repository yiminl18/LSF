def rule_net_income_row_in_income_statement_table(doc: dict) -> list[dict]:
    """Match tables containing a row labeled net income/loss in the income statement."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if not re.search(r"(statement of income|statement of operations|income statement|operations)", path + " " + text, re.I):
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "") or ""))
            for r, vals in row_text.items():
                vals_sorted = [t for _, t in sorted(vals)]
                joined = " | ".join(vals_sorted)
                if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b|\bnet income \(loss\)\b|\bnet earnings \(loss\)\b", joined, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
