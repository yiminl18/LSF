def rule_page1_numeric_or_state_values_in_company_block(doc: dict) -> list[dict]:
    """Match likely value spans in the page-1 company block: state names and EIN numbers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            path = span.get("structure", {}).get("path_text", "") or ""
            if span.get("page_no") != 1 or not path or "FORM 10-" in path:
                continue
            if re.fullmatch(r"\d{2}-\d{7}", txt):
                out.append(span)
            elif re.fullmatch(r"[A-Z][A-Za-z]+(?:\s*\([A-Za-z ]+\))?", txt) and len(txt.split()) <= 3:
                out.append(span)
        return out
    except Exception:
        return []
