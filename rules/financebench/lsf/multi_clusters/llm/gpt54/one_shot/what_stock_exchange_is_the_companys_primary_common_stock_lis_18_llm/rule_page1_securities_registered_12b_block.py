def rule_page1_securities_registered_12b_block(doc: dict) -> list[dict]:
    """Match spans on page 1 in the Section 12(b) securities-registration block."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "securities registered pursuant to section 12(b)" in txt
                or "securities registered pursuant to section 12(b) of the act" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
