import re


def rule_balance_sheet_longterm_debt(doc: dict) -> list[dict]:
    '''Consolidated balance sheet tables that carry both a Total current liabilities and a Long-term debt row.'''
    pat_ltd = re.compile(r"long-?term\s+debt", re.IGNORECASE)
    pat_tcl = re.compile(r"total\s+current\s+liabilities", re.IGNORECASE)
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        text = span.get("text", "") or ""
        if not pat_tcl.search(text):
            continue
        if not pat_ltd.search(text):
            continue
        out.append(span)
    return out
