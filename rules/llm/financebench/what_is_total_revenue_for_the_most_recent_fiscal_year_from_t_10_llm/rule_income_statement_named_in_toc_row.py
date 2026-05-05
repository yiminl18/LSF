def rule_income_statement_named_in_toc_row(doc: dict) -> list[dict]:
    """Match TOC/index tables that explicitly name the income statement page."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = ((span.get("text") or "") + " " + ((span.get("structure", {}) or {}).get("path_text", "") or "")).lower()
            if "table of contents" in txt or "index" in txt:
                if any(k in txt for k in [
                    "consolidated statement of income",
                    "consolidated statements of income",
                    "consolidated statement of operations",
                    "consolidated statements of operations",
                    "consolidated statement of earnings",
                    "consolidated statements of earnings"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
