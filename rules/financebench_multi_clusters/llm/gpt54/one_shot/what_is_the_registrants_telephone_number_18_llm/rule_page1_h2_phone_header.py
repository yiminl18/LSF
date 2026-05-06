def rule_page1_h2_phone_header(doc: dict) -> list[dict]:
    """Match H2 page-1 headers whose text is itself the phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"^\s*(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H2":
                if phone_re.match((span.get("text", "") or "").strip()):
                    out.append(span)
        return out
    except Exception:
        return []
