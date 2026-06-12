def rule_profile_of_economy_employment_and_unemployment_section(doc: dict) -> list[dict]:
    """Match text spans under the 'Employment and unemployment' subsection."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if "Employment and unemployment" in path:
                out.append(span)
    except Exception:
        return []
    return out
