def rule_page1_section12b_symbol_exchange_table(doc: dict) -> list[dict]:
    """Match page-1 Section 12(b) tables that contain trading symbols and listing exchanges."""
    try:
        import re

        trading_re = re.compile(r"trading symbol", re.IGNORECASE)
        exchange_re = re.compile(r"(?:name of each )?exchange on which registered|name of each exchange", re.IGNORECASE)
        class_re = re.compile(r"title of each class", re.IGNORECASE)

        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "table"
            and trading_re.search(" ".join((span.get("text") or "").split()))
            and exchange_re.search(" ".join((span.get("text") or "").split()))
            and class_re.search(" ".join((span.get("text") or "").split()))
        ]
    except Exception:
        return []
