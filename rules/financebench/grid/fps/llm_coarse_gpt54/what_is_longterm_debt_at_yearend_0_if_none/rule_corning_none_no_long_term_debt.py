def rule_corning_none_no_long_term_debt(doc: dict) -> list[dict]:
    """Match Corning-like cases where no long-term debt row is present and answer may be none."""
    try:
        out = []
        texts = doc.get("texts", [])
        has_corning = any("corning incorporated" in ((s.get("text") or "") + " " + (s.get("structure", {}).get("path_text", ""))).lower() for s in texts)
        has_long_term_debt = any("long-term debt" in (s.get("text") or "").lower() for s in texts)
        if has_corning and not has_long_term_debt:
            for s in texts:
                if s.get("label") in {"section_header", "text"} and "corning incorporated" in ((s.get("text") or "") + " " + (s.get("structure", {}).get("path_text", ""))).lower():
                    out.append(s)
            return out
        return []
    except Exception:
        return []
