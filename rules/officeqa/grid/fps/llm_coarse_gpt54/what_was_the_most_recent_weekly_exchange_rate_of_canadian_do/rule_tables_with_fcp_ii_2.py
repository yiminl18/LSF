def rule_tables_with_fcp_ii_2(doc: dict) -> list[dict]:
    """Match tables containing the weekly bank positions row for Canadian dollars (FCP-II-2)."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "fcp-ii-2" in txt or "fcp-1i-2" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
