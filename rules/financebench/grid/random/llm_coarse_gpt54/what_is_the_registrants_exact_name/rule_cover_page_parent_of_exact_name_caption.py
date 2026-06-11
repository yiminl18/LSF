def rule_cover_page_parent_of_exact_name_caption(doc: dict) -> list[dict]:
    """Match the cover-page parent header identified by child path_text around the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for cap in texts:
            if "exact name of registrant as specified in its charter" in ((cap.get("text") or "").lower()):
                path = cap.get("structure", {}).get("path_text") or ""
                for span in texts:
                    if (
                        span.get("page_no") == cap.get("page_no")
                        and span.get("label") == "section_header"
                        and (span.get("text") or "") == path
                    ):
                        out.append(span)
        return out
    except Exception:
        return []
