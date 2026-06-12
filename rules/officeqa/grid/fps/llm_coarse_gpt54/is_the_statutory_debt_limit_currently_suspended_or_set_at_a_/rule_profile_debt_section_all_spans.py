def rule_profile_debt_section_all_spans(doc: dict) -> list[dict]:
    """Return all spans under the Federal Budget and Debt path_text subsection."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "federal budget and debt" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
