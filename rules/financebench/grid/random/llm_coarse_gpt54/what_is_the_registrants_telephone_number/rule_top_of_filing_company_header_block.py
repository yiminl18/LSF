def rule_top_of_filing_company_header_block(doc: dict) -> list[dict]:
    """Match top-of-document company identity block spans on page 1 under the company H1."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if span.get("page_no") == 1 and path and "FORM 10-" not in path and span.get("label") in {"text", "section_header"}:
                if (span.get("structure", {}) or {}).get("depth", 99) <= 3:
                    out.append(span)
        return out
    except Exception:
        return []
