def rule_mda_results_of_operations_net_income_text(doc: dict) -> list[dict]:
    """Match text spans in MD&A / Results of Operations mentioning net income or net earnings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header"}:
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            txt = span.get("text", "") or ""
            if not re.search(r"(management'?s discussion|results of operations|financial condition)", path, re.I):
                continue
            if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
