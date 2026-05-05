def rule_balance_sheet_pages_30_to_70(doc: dict) -> list[dict]:
    """Match tables on common audited financial statement page ranges in 10-Ks."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and 30 <= int(span.get("page_no", -1)) <= 70:
                text = (span.get("text") or "").lower()
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                if "assets" in text or "balance sheet" in text or "financial statements" in path:
                    out.append(span)
        return out
    except Exception:
        return []
