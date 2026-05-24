def rule_net_sales_summary_table(doc: dict) -> list[dict]:
    """MD&A Results-of-Operations 'Net Sales' / 'Net Revenues' focused subtable. Last path component is exactly the subheading. Picks up the Costco-style Net Sales summary so the QA reads the merchandise-only revenue rather than the broader 'Total revenue' line."""
    targets = {"Net Sales", "Net sales", "Net Revenues", "Net revenues", "NET SALES", "NET REVENUES"}
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path = (span.get("structure") or {}).get("path_text", "") or ""
        parts = [p.strip() for p in path.split("|")]
        if not parts:
            continue
        if parts[-1] in targets:
            out.append(span)
    return out
