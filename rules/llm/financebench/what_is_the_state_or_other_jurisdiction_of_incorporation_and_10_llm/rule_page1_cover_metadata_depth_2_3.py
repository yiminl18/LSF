def rule_page1_cover_metadata_depth_2_3(doc: dict) -> list[dict]:
    """Match page-1 spans at structural depth 2-3 in the cover metadata area."""
    try:
        out = []
        for span in doc.get("texts", []):
            depth = span.get("structure", {}).get("depth")
            path = (span.get("structure", {}).get("path_text") or "").lower()
            if span.get("page_no") == 1 and depth in (2, 3):
                if "commission" not in path and "form 10-k" not in path:
                    out.append(span)
        return out
    except Exception:
        return []
