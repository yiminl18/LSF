def rule_mda_long_term_debt_narrative(doc: dict) -> list[dict]:
    """Match narrative spans mentioning long-term debt at year-end in MD&A or overview sections."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if ("overview" in path or "financial condition" in path or "liquidity" in path or "management" in path):
                if re.search(r"\blong[- ]term debt\b", txt, re.I) and re.search(r"\bas of\b|\bat year[- ]end\b|\bend of\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
