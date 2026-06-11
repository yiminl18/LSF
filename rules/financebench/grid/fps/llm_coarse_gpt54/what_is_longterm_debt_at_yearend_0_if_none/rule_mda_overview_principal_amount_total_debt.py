def rule_mda_overview_principal_amount_total_debt(doc: dict) -> list[dict]:
    """Match narrative spans in MD&A/Overview stating principal amount of total debt at year-end."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if "overview" in path or "management" in path or "financial condition" in path:
                if re.search(r"principal amount of total debt", txt, re.I):
                    out.append(span)
                elif re.search(r"\breduced our debt\b", txt, re.I) and re.search(r"\bdebt\b.*\bas of\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
