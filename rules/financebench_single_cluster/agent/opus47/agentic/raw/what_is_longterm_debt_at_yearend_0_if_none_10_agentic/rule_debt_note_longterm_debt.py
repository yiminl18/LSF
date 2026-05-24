import re


def rule_debt_note_longterm_debt(doc: dict) -> list[dict]:
    '''Debt-note or selected-financial-data tables that summarize long-term debt outside the main balance sheet.'''
    pat_ltd = re.compile(r"long-?term\s+debt", re.IGNORECASE)
    pat_total = re.compile(r"\btotal\b", re.IGNORECASE)
    pat_tcl = re.compile(r"total\s+current\s+liabilities", re.IGNORECASE)
    path_keywords = (
        "debt",
        "borrowings",
        "selected financial data",
        "five-year",
        "five year",
    )
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        text = span.get("text", "") or ""
        if not pat_ltd.search(text):
            continue
        if not pat_total.search(text):
            continue
        if pat_tcl.search(text):
            continue
        path = ((span.get("structure") or {}).get("path_text") or "").lower()
        if not any(kw in path for kw in path_keywords):
            continue
        out.append(span)
    return out
