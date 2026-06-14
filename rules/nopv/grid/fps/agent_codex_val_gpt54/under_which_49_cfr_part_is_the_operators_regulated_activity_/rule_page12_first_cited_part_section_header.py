def rule_page12_first_cited_part_section_header(doc: dict) -> list[dict]:
    """Match the first page-1/2 section header that cites a 49 CFR section from Part 191-199."""
    try:
        import re

        cite_re = re.compile(r"^\s*(?:\d+\s*[\.\)]\s*)?(?:[§s]\s*)?19\d\.\d+\b", re.IGNORECASE)
        for span in doc.get("texts", []):
            if span.get("page_no", 99) > 2:
                continue
            if span.get("label") != "section_header":
                continue
            if cite_re.search((span.get("text") or "").strip()):
                return [span]
        return []
    except Exception:
        return []
