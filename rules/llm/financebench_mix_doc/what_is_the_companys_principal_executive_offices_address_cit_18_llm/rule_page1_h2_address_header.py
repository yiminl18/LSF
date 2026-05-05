def rule_page1_h2_address_header(doc: dict) -> list[dict]:
    """Match page 1 H2 section headers that are themselves street-address headers."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            if span.get("structure", {}).get("level") != "H2":
                continue
            txt = span.get("text") or ""
            if re.search(r"^\d{1,5}\s", txt):
                out.append(span)
            elif "warmley" in txt.lower() or "bristol" in txt.lower():
                out.append(span)
        return out
    except Exception:
        return []
