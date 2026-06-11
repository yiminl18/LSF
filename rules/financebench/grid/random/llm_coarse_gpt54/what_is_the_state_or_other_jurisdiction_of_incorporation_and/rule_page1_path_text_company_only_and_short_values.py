def rule_page1_path_text_company_only_and_short_values(doc: dict) -> list[dict]:
    """Match short value-like spans whose path_text is just the company name."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") != 1 or "|" in path or not path:
                continue
            if re.fullmatch(r"\d{2}-\d{7}", txt) or re.fullmatch(r"[A-Z][A-Za-z]+(?:\s*\([A-Za-z ]+\))?", txt):
                out.append(span)
        return out
    except Exception:
        return []
