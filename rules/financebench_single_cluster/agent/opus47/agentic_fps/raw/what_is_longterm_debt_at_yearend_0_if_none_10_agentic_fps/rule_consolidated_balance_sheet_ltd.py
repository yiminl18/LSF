import re


def rule_consolidated_balance_sheet_ltd(doc: dict) -> list[dict]:
    """Consolidated balance sheet table — contains a long-term/term debt line plus Assets and Liabilities sections."""
    pat_debt = re.compile(r"(long-?term|term) debt", re.IGNORECASE)
    pat_assets = re.compile(r"\bassets\b", re.IGNORECASE)
    pat_liab = re.compile(r"\bliabilities\b", re.IGNORECASE)
    out = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        text = span.get("text", "") or ""
        if pat_debt.search(text) and pat_assets.search(text) and pat_liab.search(text):
            out.append(span)
    return out
