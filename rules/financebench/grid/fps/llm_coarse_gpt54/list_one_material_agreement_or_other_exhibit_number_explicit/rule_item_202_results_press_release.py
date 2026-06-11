def rule_item_202_results_press_release(doc: dict) -> list[dict]:
    """Match Item 2.02 / 7.01 sections that often point to Exhibit 99.1 press releases."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            blob = path + " " + text
            if re.search(r'item\s*2\.02|item\s*7\.01', blob, re.I) and re.search(r'press release|exhibit\s*99', blob, re.I):
                out.append(span)
    except Exception:
        return []
    return out
