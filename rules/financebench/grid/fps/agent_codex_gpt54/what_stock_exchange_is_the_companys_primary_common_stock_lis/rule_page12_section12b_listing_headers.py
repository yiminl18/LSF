def rule_page12_section12b_listing_headers(doc: dict) -> list[dict]:
    """Match page-1/2 Section 12(b) header and label spans that frame the listing answer."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            lowered = span.get("text", "").lower()
            if "indicate by check mark" in lowered:
                continue

            if (
                "section 12(b)" in lowered
                or "trading symbol" in lowered
                or "name of each exchange on which registered" in lowered
                or "name of exchange on which registered" in lowered
            ):
                results.append(span)

        return results
    except Exception:
        return []
