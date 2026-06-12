def rule_esf_assets_liabilities_table(doc: dict) -> list[dict]:
    """Match ESF table spans that look like a balance sheet with assets/liabilities terminology."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            combo = f"{path}\n{text}"
            if re.search(r'Exchange Stabilization Fund|ESF-?1', combo, re.I) and (
                re.search(r'assets', combo, re.I) or re.search(r'liabilities', combo, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
