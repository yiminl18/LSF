def rule_page12_common_stock_spans_near_section12b(doc: dict) -> list[dict]:
    """Match common-stock title spans that sit inside the page-1/2 Section 12(b) listing block."""
    try:
        import re

        texts = doc.get("texts", [])
        results = []

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            lowered = span.get("text", "").lower()
            if (
                "aggregate market value" in lowered
                or "shares outstanding" in lowered
                or "number of shares outstanding" in lowered
            ):
                continue

            if not (
                "common stock" in lowered
                or "ordinary shares" in lowered
                or re.search(r"class\s+[a-z]\s+common stock", lowered)
            ):
                continue

            start = max(0, i - 8)
            end = min(len(texts), i + 8)
            window = " ".join(
                texts[j].get("text", "").lower()
                for j in range(start, end)
            )
            if any(
                marker in window
                for marker in (
                    "section 12(b)",
                    "trading symbol",
                    "exchange on which registered",
                )
            ):
                results.append(span)

        return results
    except Exception:
        return []
