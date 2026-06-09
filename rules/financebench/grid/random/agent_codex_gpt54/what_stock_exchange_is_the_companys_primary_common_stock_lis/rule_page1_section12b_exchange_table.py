def rule_page1_section12b_exchange_table(doc: dict) -> list[dict]:
    """Match page-1 Section 12(b) tables that include the exchange registration row."""
    try:
        import re

        registered_re = re.compile(r"name of (?:each )?exchange on which registered", re.IGNORECASE)
        common_re = re.compile(r"\b(?:common stock|ordinary shares?)\b", re.IGNORECASE)
        exchange_re = re.compile(
            r"\b(?:new york stock exchange|nyse|nasdaq(?: global select market| global market| capital market)?)\b",
            re.IGNORECASE,
        )

        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "table"
            and registered_re.search(" ".join((span.get("text") or "").split()))
            and common_re.search(" ".join((span.get("text") or "").split()))
            and exchange_re.search(" ".join((span.get("text") or "").split()))
        ]
    except Exception:
        return []
