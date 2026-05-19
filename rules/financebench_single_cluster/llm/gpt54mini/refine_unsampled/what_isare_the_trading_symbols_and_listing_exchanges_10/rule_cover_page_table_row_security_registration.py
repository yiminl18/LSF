def rule_cover_page_table_row_security_registration(doc: dict) -> list[dict]:
    """Match table spans whose cells contain Section 12(b) registration row content."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            cell_text = " ".join((c.get("text") or "") for c in cells).lower()
            if "section 12(b)" in cell_text or ("trading symbol" in cell_text and "exchange" in cell_text):
                out.append(span)
        return out
    except Exception:
        return []
