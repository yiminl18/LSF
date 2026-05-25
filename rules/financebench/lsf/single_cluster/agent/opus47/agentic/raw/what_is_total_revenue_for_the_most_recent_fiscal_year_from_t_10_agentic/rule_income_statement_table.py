import re


def rule_income_statement_table(doc: dict) -> list[dict]:
    """Audited consolidated income statement table, identified by 'Consolidated Statement(s) of Income/Operations/Earnings' in path_text or table body. Skips tables that combine 'Membership fees' with a 'Total revenue' line (e.g., Costco), where the labelled answer follows the Net sales line — those are covered by a sibling Net Sales rule."""
    title_re = re.compile(
        r"consolidated\s+statements?\s+of\s+(income|operations|earnings)",
        re.IGNORECASE,
    )
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path = (span.get("structure") or {}).get("path_text", "") or ""
        text = span.get("text", "") or ""
        if "Membership fees" in text and "Total revenue" in text:
            continue
        if title_re.search(path):
            out.append(span)
        elif title_re.search(text) and re.search(r"net\s+income|net\s+earnings", text, re.IGNORECASE):
            out.append(span)
    return out
