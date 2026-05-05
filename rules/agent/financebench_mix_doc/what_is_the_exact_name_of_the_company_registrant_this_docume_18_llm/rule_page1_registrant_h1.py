def rule_page1_registrant_h1(doc: dict) -> list[dict]:
    """Retrieve the page-1 H1 registrant name header while excluding generic filing headers."""
    try:
        import re
        spans = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if not text:
                continue
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            structure = span.get("structure") or {}
            if structure.get("level") != "H1":
                continue
            upper = text.upper()
            if upper in {
                "FORM 10-K", "FORM 10-Q", "FORM 8-K", "CURRENT REPORT",
                "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
                "SECURITIES AND EXCHANGE COMMISSION"
            }:
                continue
            if "SECURITIES AND EXCHANGE COMMISSION" in upper:
                continue
            if upper.startswith("FORM "):
                continue
            if upper.startswith("CURRENT REPORT"):
                continue
            if len(re.findall(r"[A-Z0-9]", upper)) < 3:
                continue
            spans.append(span)
        return spans
    except Exception:
        return []

