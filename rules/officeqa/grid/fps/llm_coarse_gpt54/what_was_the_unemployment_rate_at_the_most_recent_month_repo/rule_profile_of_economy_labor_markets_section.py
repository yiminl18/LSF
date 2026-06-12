def rule_profile_of_economy_labor_markets_section(doc: dict) -> list[dict]:
    """Match text spans under the 'Labor Markets' subsection."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Labor Markets" in path:
                out.append(span)
    except Exception:
        return []
    return out
