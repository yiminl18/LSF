def rule_company_name_with_corporate_suffix(doc: dict) -> list[dict]:
    """Match page-1 prominent spans containing common corporate suffixes like Inc., Corporation, plc, or Company."""
    try:
        texts = doc.get("texts", [])
        out = []
        suffixes = ["inc.", "inc", "corporation", "company", "plc", "incorporated", "co.,", "co ", "corp"]
        for span in texts:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if float(span.get("size") or 0) < 9:
                continue
            if any(s in low for s in suffixes):
                if "exact name of registrant" not in low and "securities and exchange commission" not in low:
                    out.append(span)
        return out
    except Exception:
        return []
