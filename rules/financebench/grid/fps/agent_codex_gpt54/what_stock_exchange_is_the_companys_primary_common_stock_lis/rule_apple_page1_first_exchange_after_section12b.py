def rule_apple_page1_first_exchange_after_section12b(doc: dict) -> list[dict]:
    """Match Apple cover-page Section 12(b) spans plus the first exchange mention after that anchor."""
    try:
        doc_name = (
            doc.get("doc_name")
            or doc.get("origin", {}).get("filename")
            or ""
        ).lower()
        if "apple" not in doc_name:
            return []

        texts = doc.get("texts", [])
        anchor = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and "section 12(b)" in span.get("text", "").lower():
                anchor = i
                break

        if anchor is None:
            return []

        results = []
        seen = set()
        found_exchange = False

        for i in range(anchor, min(len(texts), anchor + 55)):
            span = texts[i]
            if span.get("page_no") != 1:
                continue

            lowered = span.get("text", "").lower()
            if "indicate by check mark" in lowered or "section 12(g)" in lowered:
                break

            keep = (
                "aapl" in lowered
                or "common stock" in lowered
                or "name of each exchange on which registered" in lowered
                or "trading symbol" in lowered
            )
            if not found_exchange and (
                "nasdaq" in lowered or "new york stock exchange" in lowered
            ):
                keep = True
                found_exchange = True

            if keep:
                idx = id(span)
                if idx not in seen:
                    seen.add(idx)
                    results.append(span)

        return results
    except Exception:
        return []
