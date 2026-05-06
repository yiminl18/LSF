def rule_page1_registrant_h1(doc: dict) -> list[dict]:
    """Retrieve the page-1 H1 registrant-name header on the cover page."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            structure = span.get("structure") or {}
            if structure.get("level") != "H1":
                continue
            txt = (span.get("text") or "").strip()
            if not txt:
                continue
            upper = txt.upper()
            if upper in {
                "FORM 10-K", "FORM 10-Q", "FORM 8-K", "CURRENT REPORT",
                "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
                "SECURITIES AND EXCHANGE COMMISSION"
            }:
                continue
            if "WASHINGTON, D.C." in upper:
                continue
            if re.fullmatch(r"[\W_]+", txt):
                continue
            out.append(span)
        return out
    except Exception:
        return []

