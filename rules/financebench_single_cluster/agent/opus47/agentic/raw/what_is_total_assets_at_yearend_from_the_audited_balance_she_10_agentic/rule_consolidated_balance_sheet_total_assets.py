import re


def rule_consolidated_balance_sheet_total_assets(doc: dict) -> list[dict]:
    """Audited consolidated balance sheet / statement-of-financial-position table that contains the Total-assets row, identified either by the section path or by the table's own header line."""
    keywords = (
        "consolidated balance sheet",
        "consolidated statement of financial position",
        "consolidated statements of financial position",
    )
    total_assets_row = re.compile(r"\|\s*total\s+assets\s*\|", re.IGNORECASE)
    out = []
    for span in doc.get("texts", []) or []:
        if span.get("label") != "table":
            continue
        text = span.get("text") or ""
        if not total_assets_row.search(text):
            continue
        path = ((span.get("structure") or {}).get("path_text") or "").lower()
        header_block = "\n".join(text.splitlines()[:3]).lower()
        if any(k in path for k in keywords) or any(k in header_block for k in keywords):
            out.append(span)
    return out
