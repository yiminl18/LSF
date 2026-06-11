def rule_reports_financial_results_title(doc: dict) -> list[dict]:
    """Match page-1 titles containing 'Reports ... Financial Results', a common earnings-release pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"reports .*financial results", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
