def rule_profile_of_economy_table_of_indicators(doc: dict) -> list[dict]:
    """Match tables in Profile of the Economy that likely summarize economic indicators including sentiment readings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "profile of the economy" in path.lower():
                if re.search(r"(economic indicators|consumer sentiment|consumer confidence|michigan|reuters)", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
