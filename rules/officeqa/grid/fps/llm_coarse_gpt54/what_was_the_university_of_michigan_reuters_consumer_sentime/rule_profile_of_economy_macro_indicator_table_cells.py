def rule_profile_of_economy_macro_indicator_table_cells(doc: dict) -> list[dict]:
    """Match tables/cells in Profile of the Economy that may contain macro indicator readings including sentiment."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "profile of the economy" not in path.lower():
                continue
            text = (span.get("text") or "")
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if re.search(r"(sentiment|confidence|michigan|reuters|economic indicators)", text, re.I):
                out.append(span)
                continue
            for c in cells:
                ct = c.get("text") or ""
                if re.search(r"(sentiment|confidence|michigan|reuters)", ct, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
