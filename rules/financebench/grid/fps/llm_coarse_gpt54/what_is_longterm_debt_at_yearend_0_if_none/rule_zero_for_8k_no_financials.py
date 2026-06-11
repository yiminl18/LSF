def rule_zero_for_8k_no_financials(doc: dict) -> list[dict]:
    """Match 8-K documents with no financial statements or only event/exhibit content, suggesting answer 0."""
    try:
        out = []
        texts = doc.get("texts", [])
        has_8k = any("form 8-k" in (s.get("text") or "").lower() or "form 8-k" in (s.get("structure", {}).get("path_text", "") or "").lower() for s in texts)
        has_balance_sheet = any("balance sheet" in ((s.get("text") or "") + " " + (s.get("structure", {}).get("path_text", "") or "")).lower() for s in texts)
        has_debt = any("long-term debt" in (s.get("text") or "").lower() for s in texts)
        if has_8k and not has_balance_sheet and not has_debt:
            for s in texts:
                if s.get("label") in {"section_header", "text"} and ("form 8-k" in (s.get("text") or "").lower() or "current report" in (s.get("text") or "").lower()):
                    out.append(s)
            return out
        return []
    except Exception:
        return []
