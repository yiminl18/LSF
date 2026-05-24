def rule_tables_after_balance_sheet_heading(doc: dict) -> list[dict]:
    """Match tables appearing shortly after a balance-sheet-related heading."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            window = texts[max(0, i - 5):i]
            joined = " ".join((w.get("text") or "").lower() for w in window)
            if "balance sheet" in joined or "statement of financial position" in joined:
                out.append(span)
        return out
    except Exception:
        return []
