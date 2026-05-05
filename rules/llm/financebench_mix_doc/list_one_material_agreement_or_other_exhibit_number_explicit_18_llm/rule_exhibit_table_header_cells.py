def rule_exhibit_table_header_cells(doc: dict) -> list[dict]:
    """Match tables whose header cells include exhibit-related columns like Exhibit No. or Description."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            header_texts = [((c.get("text") or "").strip()) for c in cells if c.get("is_column_header")]
            joined = " | ".join(header_texts)
            if re.search(r"exhibit no\.?", joined, re.I) or (
                re.search(r"\bdescription\b", joined, re.I) and re.search(r"\bexhibit\b", (span.get("text") or ""), re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
