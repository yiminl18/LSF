def rule_page1_cover_identification_tables_or_text(doc: dict) -> list[dict]:
    """Match page-1 cover-page identification spans or tables likely containing the incorporation state and EIN."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("label") == "table":
                cells = ((span.get("table_data") or {}).get("cells") or [])
                txt += " " + " ".join((c.get("text") or "") for c in cells)
            if re.search(r"exact name of registrant|state\s+or\s+other\s+jurisdiction|employer\s+identification", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
