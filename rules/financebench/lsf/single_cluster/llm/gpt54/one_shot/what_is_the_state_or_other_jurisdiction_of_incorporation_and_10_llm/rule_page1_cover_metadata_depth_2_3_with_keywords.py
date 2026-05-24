def rule_page1_cover_metadata_depth_2_3_with_keywords(doc: dict) -> list[dict]:
    """Match page-1 depth-2/3 cover metadata spans containing state/EIN keywords or values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            depth = span.get("structure", {}).get("depth")
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and depth in (2, 3):
                if re.search(r"\d{2}-\d{7}|state|jurisdiction|incorporation|employer identification", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
