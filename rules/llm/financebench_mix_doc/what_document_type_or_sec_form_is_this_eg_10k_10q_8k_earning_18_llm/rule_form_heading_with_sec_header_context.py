def rule_form_heading_with_sec_header_context(doc: dict) -> list[dict]:
    """Match form/report spans on pages that also contain the SEC commission header."""
    import re
    try:
        pages = set()
        for s in doc.get("texts", []):
            if "SECURITIES AND EXCHANGE COMMISSION" in ((s.get("text") or "").upper()):
                pages.add(s.get("page_no"))
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") in pages
            and (
                re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (s.get("text") or ""), re.I)
                or "CURRENT REPORT" in ((s.get("text") or "").upper())
            )
        ]
    except Exception:
        return []
