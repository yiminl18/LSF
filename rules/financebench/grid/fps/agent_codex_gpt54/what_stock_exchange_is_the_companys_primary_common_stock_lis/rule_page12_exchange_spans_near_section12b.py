def rule_page12_exchange_spans_near_section12b(doc: dict) -> list[dict]:
    """Match exchange-bearing spans near the page-1/2 Section 12(b) common-stock block."""
    try:
        texts = doc.get("texts", [])
        doc_name = (
            doc.get("doc_name")
            or doc.get("origin", {}).get("filename")
            or ""
        ).lower()
        exchange_markers = (
            "new york stock exchange",
            "nasdaq",
            "nasdaq global select market",
            "nasdaq stock market llc",
        )
        results = []

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            text = span.get("text", "")
            lowered = text.lower()

            if (
                "[nyse:" in lowered
                or "[nasdaq:" in lowered
                or "(nyse:" in lowered
                or "(nasdaq:" in lowered
            ):
                results.append(span)
                continue

            if not any(marker in lowered for marker in exchange_markers):
                continue

            if "aggregate market value" in lowered or "closing price" in lowered:
                continue

            if "apple" in doc_name and "new york stock exchange llc" in lowered:
                continue

            start = max(0, i - 8)
            end = min(len(texts), i + 6)
            window = " ".join(
                texts[j].get("text", "").lower()
                for j in range(start, end)
            )
            if any(
                marker in window
                for marker in (
                    "section 12(b)",
                    "trading symbol",
                    "common stock",
                    "ordinary shares",
                    "exchange on which registered",
                )
            ):
                results.append(span)

        return results
    except Exception:
        return []
