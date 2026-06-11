def rule_page1_company_header_with_children(doc: dict) -> list[dict]:
    """Match page-1 H1 headers that have many child body spans, typical of the registrant cover block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text") or "").upper()
            ):
                path = span.get("structure", {}).get("path_text") or ""
                child_count = sum(
                    1 for s in texts
                    if s.get("page_no") in (1, 2)
                    and (s.get("structure", {}).get("path_text") or "") == path
                    and s is not span
                )
                if child_count >= 3:
                    out.append(span)
        return out
    except Exception:
        return []
