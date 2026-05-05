def rule_page1_bold_small_listing_lines(doc: dict) -> list[dict]:
    """Match bold page-1 small-font lines in the cover listing block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and float(span.get("size") or 0) <= 10.5
            ):
                txt = (span.get("text") or "").lower()
                if (
                    "trading symbol" in txt
                    or "exchange on which registered" in txt
                    or "name of each exchange" in txt
                    or "section 12(b)" in txt
                ):
                    out.append(span)
        return out
    except Exception:
        return []
