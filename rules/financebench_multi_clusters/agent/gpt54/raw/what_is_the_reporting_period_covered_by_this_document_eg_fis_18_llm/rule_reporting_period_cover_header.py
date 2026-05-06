def rule_reporting_period_cover_header(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-header spans stating the fiscal/quarterly period or event date."""
    try:
        import re
        texts = doc.get("texts", []) or []
        out = []
        for span in texts:
            if not isinstance(span, dict):
                continue
            if span.get("page_no") != 1:
                continue
            label = span.get("label")
            if label not in {"text", "section_header", "checkbox_selected"}:
                continue
            st = span.get("structure") or {}
            path = (st.get("path_text") or "")
            txt = (span.get("text") or "")
            txt_l = txt.lower()
            path_l = path.lower()
            if not (
                "form 10-k" in path_l or
                "form 10-q" in path_l or
                "form 8-k" in path_l or
                "current report" in path_l
            ):
                continue
            if (
                "fiscal year ended" in txt_l or
                "quarterly period ended" in txt_l or
                "quarter ended" in txt_l or
                "date of report" in txt_l or
                "earliest event reported" in txt_l
            ):
                out.append(span)
                continue
            # catch combined annual/quarterly report checkbox/header lines on cover page
            if re.search(r"\bannual report\b|\bquarterly report\b", txt_l) and re.search(r"ended\b", txt_l):
                out.append(span)
        return out
    except Exception:
        return []

