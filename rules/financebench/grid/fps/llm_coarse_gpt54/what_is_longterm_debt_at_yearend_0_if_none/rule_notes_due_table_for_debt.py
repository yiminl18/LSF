def rule_notes_due_table_for_debt(doc: dict) -> list[dict]:
    """Match tables listing notes due by year, often used to derive long-term debt balances."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r"notes due", txt, re.I) or re.search(r"senior notes due", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
