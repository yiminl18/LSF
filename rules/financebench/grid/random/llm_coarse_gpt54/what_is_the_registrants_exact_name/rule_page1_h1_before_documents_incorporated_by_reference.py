def rule_page1_h1_before_documents_incorporated_by_reference(doc: dict) -> list[dict]:
    """Match the page-1 H1 company header whose section continues into 'DOCUMENTS INCORPORATED BY REFERENCE'."""
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
                children = [
                    s for s in texts
                    if (s.get("structure", {}).get("path_text") or "") == path and s.get("page_no") in (1, 2)
                ]
                joined = " ".join((c.get("text") or "").upper() for c in children)
                if "DOCUMENTS INCORPORATED BY REFERENCE" in joined:
                    out.append(span)
        return out
    except Exception:
        return []
