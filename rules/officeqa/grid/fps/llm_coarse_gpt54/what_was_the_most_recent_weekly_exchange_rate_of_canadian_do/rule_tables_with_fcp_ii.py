def rule_tables_with_fcp_ii(doc: dict) -> list[dict]:
    """Match tables containing FCP-II entries, the Canadian dollar positions table family."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "fcp-ii-" in txt or "fcp ii" in txt or "fcp-1i-" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
