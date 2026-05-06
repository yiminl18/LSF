def rule_selected_financial_data_table(doc: dict) -> list[dict]:
    """Tables whose immediate parent section is a Selected Financial Data heading."""
    try:
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            path = ((sp.get("structure") or {}).get("path_text") or "").lower()
            segs = [s.strip() for s in path.split("|")]
            if not segs:
                continue
            leaf = segs[-1]
            if (
                "selected financial data" in leaf
                or "selected consolidated financial data" in leaf
                or "five-year" in leaf
            ):
                out.append(sp)
        return out
    except Exception:
        return []
