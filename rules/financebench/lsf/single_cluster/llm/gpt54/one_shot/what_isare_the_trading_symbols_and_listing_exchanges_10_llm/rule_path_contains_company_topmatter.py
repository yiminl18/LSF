def rule_path_contains_company_topmatter(doc: dict) -> list[dict]:
    """Match top-matter spans under the company name on page 1 that mention exchange/symbol registration."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and path and re.search(r"registered|exchange|symbol|12\(b\)", txt + " " + path, re.I):
                out.append(span)
        return out
    except Exception:
        return []
