def rule_contents_entry_for_target_table(doc: dict) -> list[dict]:
    """Match contents-page entries naming the target table, useful for locating the relevant page/table."""
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                "contents" in path
                and "maturity distribution" in txt
                and "average length" in txt
                and "marketable" in txt
            ):
                out.append(span)
    except Exception:
        return []
    return out
