def rule_net_income_summary_or_income_statement(doc: dict) -> list[dict]:
    """Smallest MD&A summary table (attributable-to or billions-scale Net Earnings); falls back to the consolidated income statement table."""
    import re
    discussion_re = re.compile(
        r"discussion\s+and\s+analysis|results\s+of\s+operations|consolidated\s+results|liquidity",
        re.IGNORECASE,
    )
    attr_re = re.compile(r"net\s+(income|earnings)\s+attributable\s+to", re.IGNORECASE)
    ne_re = re.compile(r"Net\s+(Earnings|Income|Loss)")
    billions_re = re.compile(r"in\s+billions", re.IGNORECASE)

    md_a_summary = None
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path = (span.get("structure") or {}).get("path_text", "") or ""
        text = span.get("text", "") or ""
        if not discussion_re.search(path) or len(text) > 1500:
            continue
        if attr_re.search(text) or (ne_re.search(text) and billions_re.search(text)):
            if md_a_summary is None or len(text) < len(md_a_summary.get("text", "")):
                md_a_summary = span

    if md_a_summary is not None:
        return [md_a_summary]

    head_re = re.compile(
        r"consolidated\s+statements?\s+of\s+(income|operations|earnings)",
        re.IGNORECASE,
    )
    body_re = re.compile(r"net\s+(income|earnings|loss)", re.IGNORECASE)
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        struct = span.get("structure") or {}
        path = struct.get("path_text", "") or ""
        text = span.get("text", "") or ""
        if (head_re.search(path) or head_re.search(text)) and body_re.search(text):
            out.append(span)
    return out
