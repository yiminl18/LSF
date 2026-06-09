def rule_page1_exchange_with_registered_header_context(doc: dict) -> list[dict]:
    """Match short exchange spans on page 1 that sit next to Section 12(b) registration headers."""
    try:
        import re

        exchange_re = re.compile(
            r"\b(?:new york stock exchange|nyse|nasdaq(?: global select market| global market| capital market)?|the nasdaq global select market|the nasdaq global market|the new york stock exchange)\b",
            re.IGNORECASE,
        )
        registered_re = re.compile(r"name of (?:each )?exchange on which registered", re.IGNORECASE)
        section12b_re = re.compile(r"securities registered pursuant to section 12\(b\)", re.IGNORECASE)
        blocked_re = re.compile(
            r"aggregate market value|holders of record|also traded on|also listed on|closing sale price|closing price|stockholders of record|the number of shares",
            re.IGNORECASE,
        )

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        page1_spans = [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1 and span.get("label") != "table"
        ]

        matched: list[dict] = []
        for idx, span in enumerate(page1_spans):
            text = normalize(span.get("text") or "")
            if not text or len(text) > 120:
                continue
            if not exchange_re.search(text) or blocked_re.search(text):
                continue

            path_text = normalize(((span.get("structure") or {}).get("path_text")) or "")
            if registered_re.search(path_text):
                matched.append(span)
                continue

            window_text = " ".join(
                normalize(page1_spans[j].get("text") or "")
                for j in range(max(0, idx - 6), idx)
            )
            if registered_re.search(window_text) or section12b_re.search(window_text):
                matched.append(span)

        return matched
    except Exception:
        return []
