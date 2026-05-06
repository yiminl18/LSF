def rule_balance_sheet_table(doc: dict) -> list[dict]:
    """Tables under a Balance Sheet / Statements of Financial Position section."""
    try:
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            path = ((sp.get("structure") or {}).get("path_text") or "").lower()
            segs = [s.strip() for s in path.split("|")]
            if any(
                seg == "balance sheet"
                or seg == "balance sheets"
                or "consolidated balance sheet" in seg
                or "statement of financial position" in seg
                or "statements of financial position" in seg
                or "key balance sheet data" in seg
                or seg.endswith("balance sheets")
                or seg.endswith("balance sheet")
                for seg in segs
            ):
                out.append(sp)
        return out
    except Exception:
        return []
