def rule_esf_total_assets_row(doc: dict) -> list[dict]:
    """Match ESF table spans containing a Total assets row/cell."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            has_total_assets = re.search(r'total assets', text, re.I) is not None
            if not has_total_assets:
                for c in cells:
                    if re.search(r'total assets', c.get("text", "") or "", re.I):
                        has_total_assets = True
                        break
            if has_total_assets and re.search(r'ESF|Exchange Stabilization Fund', path + " " + text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
